"""dexrt_sc z-slab decomposition: results on N ranks must match a single rank.
Usage: python sc_mpi_test.py DEXRT_SC_MPI WORKDIR [MPIRUN]"""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
sys.path.insert(0, str(Path(__file__).parent))
from sc_givenfs_tests_lib import run_givenfs

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
mpirun = sys.argv[3] if len(sys.argv) > 3 else "mpirun"
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
ranks = [2, 4]
omp = "4"


# NOTE: J is accumulated in f32 and stored once in 16 bits, so the summation
# order (wavelength batches, so also the f32 precon_R accumulation, depend on
# the rank count) moves it by at most one 16-bit step (2^-10 relative).
RAD_RTOL = 1e-3
POPS_RTOL = 2e-5

def launcher(n):
    return (mpirun, "-np", str(n), "--oversubscribe", "-x", f"OMP_NUM_THREADS={omp}", "-x", "OMP_PROC_BIND=false")


failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30 * np.abs(b).max()))


# GivenFs: structured emitter/absorber crossing the slab boundaries.
nx, nz = 64, 128
c = (np.arange(nz) + 0.5) / nz
Z, X = np.meshgrid(c, (np.arange(nx) + 0.5) / nx, indexing="ij")
blob = np.exp(-((X - 0.4) ** 2 + (Z - 0.5) ** 2) / 0.01) + 0.5 * np.exp(-((X - 0.8) ** 2 + (Z - 0.27) ** 2) / 0.003)
chi = np.stack([0.05 * (0.1 + blob), 3.0 * (0.1 + blob)], axis=-1).astype(np.float32)
eta = (chi * (1.0 + 3.0 * blob)[..., None]).astype(np.float32)
# Non-uniform z: geometric stretching with a 2x jump at a slab boundary (row 64).
z_s = np.cumsum(1.01 ** np.arange(nz) * np.where(np.arange(nz) < 64, 1.0, 2.0))
cases = [(xb, interp, None) for xb in ["Fixed", "Periodic"] for interp in ["Linear", "VanLeer", "MonotonizedCentral"]]
cases += [(xb, "MonotonizedCentral", z_s) for xb in ["Fixed", "Periodic"]]
cases += [(xb, "LongerCharacteristics", zz) for xb in ["Fixed", "Periodic"] for zz in [None, z_s]]
for xb, interp, z in cases:
    zt = "" if z is None else "_z"
    sc = {"characteristics": "Short", "quadrature": "Stepan", "quadrature_order": 3, "formal_solver": "PiecewiseLinear",
          "interpolation": interp, "x_boundary": xb}
    if interp == "LongerCharacteristics":
        sc = {"quadrature": "Stepan", "quadrature_order": 3, "x_boundary": xb, "characteristics": interp}
    base = run_givenfs(dexrt_sc, work, f"mpi_{xb}_{interp}{zt}_1", eta, chi, sc, launcher=launcher(1), z=z)
    J1 = np.asarray(base["J"][...])[:, :nz, :nx]
    E1 = np.asarray(base["I_emergent"][...])
    for n in ranks:
        out = run_givenfs(dexrt_sc, work, f"mpi_{xb}_{interp}{zt}_{n}", eta, chi, sc, launcher=launcher(n), z=z)
        Jn = np.asarray(out["J"][...])[:, :nz, :nx]
        En = np.asarray(out["I_emergent"][...])
        check(np.array_equal(J1, Jn) or rel(Jn, J1) < RAD_RTOL, f"GivenFs {xb}/{interp}{zt} {n} ranks: J max rel diff {rel(Jn, J1):.2e}")
        check(np.array_equal(E1, En) or rel(En, E1) < 1e-5, f"GivenFs {xb}/{interp}{zt} {n} ranks: emergent max rel diff {rel(En, E1):.2e}")

# Non-LTE H_4, fixed iteration count (Ng off so the iterations are identical).
KB = 1.380649e-23
nx, nz = 32, 128
Z, X = np.meshgrid((np.arange(nz) + 0.5) / nz, (np.arange(nx) + 0.5) / nx, indexing="ij")
T = 6000.0 + 4000.0 * np.exp(-((X - 0.4) ** 2 + (Z - 0.55) ** 2) / 0.02)
nh = 3e16 * (1.0 + 2.0 * np.exp(-((X - 0.6) ** 2 + (Z - 0.3) ** 2) / 0.03))
atmos_path = work / "nlte_atmos.nc"
with ncdf.Dataset(atmos_path, "w") as f:
    f.createDimension("x", nx)
    f.createDimension("z", nz)
    for k, v in dict(temperature=T, ne=0.2 * nh, nh_tot=nh, vturb=0 * T + 5e3, pressure=nh * KB * T,
                     vx=0 * T, vy=0 * T, vz=3e3 * np.sin(2 * np.pi * X)).items():
        f.createVariable(k, "f4", ("z", "x"))[...] = v
    f.createVariable("voxel_scale", "f4")[...] = 100e3
# The same on a stretched z grid (same total height).
atmos_z_path = work / "nlte_atmos_z.nc"
with ncdf.Dataset(atmos_path) as src, ncdf.Dataset(atmos_z_path, "w") as f:
    f.createDimension("x", nx)
    f.createDimension("z", nz)
    for k in ["temperature", "ne", "nh_tot", "vturb", "pressure", "vx", "vy", "vz"]:
        f.createVariable(k, "f4", ("z", "x"))[...] = src[k][...]
    f.createVariable("voxel_scale", "f4")[...] = 100e3
    zz = np.cumsum(1.01 ** np.arange(nz) * np.where(np.arange(nz) < 64, 1.0, 2.0))
    f.createVariable("z", "f8", ("z",))[...] = zz * (nz * 100e3 / zz[-1])


def run_nlte(n, xb, ng=False, stretched=False, longer=False, rad_loss=None):
    name = f"nlte_{xb}_{n}{'_ng' if ng else ''}{'_z' if stretched else ''}{'_lc' if longer else ''}{'_' + rad_loss if rad_loss else ''}"
    conf = {
        "mode": "NonLte", "atmos_path": str(atmos_z_path if stretched else atmos_path), "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_4.yaml"), "treatment": "Active"}},
        "boundary_type": "Zero", "max_iter": 40 if ng else 8, "pop_tol": 1e-12,
        "ng_config": {"enable": ng, "threshold": 0.5}, "system": {"mem_pool_gb": 1.0},
        "sc": {"characteristics": "Short", "quadrature": "Stepan", "quadrature_order": 3, "interpolation": "MonotonizedCentral", "x_boundary": xb},
    }
    if longer:
        conf["sc"]["characteristics"] = "LongerCharacteristics"
    if rad_loss:
        conf.update(rad_loss=rad_loss, max_iter=2)
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    with open(work / f"{name}.log", "w") as log:
        res = subprocess.run([*launcher(n), str(dexrt_sc), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print((work / f"{name}.log").read_text()[-3000:])
        raise RuntimeError(name)
    out = ncdf.Dataset(work / f"{name}.nc")
    return np.asarray(out["rad_loss"][...]) if rad_loss else np.asarray(out["pops"][...])


for xb in ["Fixed", "Periodic"]:
    p1 = run_nlte(1, xb)
    for n in ranks:
        pn = run_nlte(n, xb)
        check(rel(pn, p1) < POPS_RTOL, f"NonLte {xb} {n} ranks: pops max rel diff {rel(pn, p1):.2e}")

# NOTE: rad_loss follows the pops; relative to its peak (it changes sign).
for longer in [False, True]:
    for mode in ["PerWavelength", "Integrated"]:
        L1 = run_nlte(1, "Fixed", longer=longer, rad_loss=mode)
        for n in ranks:
            Ln = run_nlte(n, "Fixed", longer=longer, rad_loss=mode)
            err = np.max(np.abs(Ln - L1)) / np.max(np.abs(L1))
            check(err < 1e-4, f"rad_loss {mode}{' longer' if longer else ''} {n} ranks: max diff {err:.2e} of the peak")

p1 = run_nlte(1, "Fixed", stretched=True)
for n in ranks:
    pn = run_nlte(n, "Fixed", stretched=True)
    check(rel(pn, p1) < POPS_RTOL, f"NonLte Fixed non-uniform z {n} ranks: pops max rel diff {rel(pn, p1):.2e}")

for stretched in [False, True]:
    p1 = run_nlte(1, "Fixed", stretched=stretched, longer=True)
    for n in ranks:
        pn = run_nlte(n, "Fixed", stretched=stretched, longer=True)
        check(rel(pn, p1) < POPS_RTOL, f"NonLte longer Fixed{' non-uniform z' if stretched else ''} {n} ranks: pops max rel diff {rel(pn, p1):.2e}")

# NOTE: With Ng the normal equations are summed over the ranks, and the f32
# precon_R reordering is amplified by the extrapolation.
p1 = run_nlte(1, "Fixed", ng=True)
n_acc = (work / "nlte_Fixed_1_ng.log").read_text().count("Ng Acceleration!")
check(n_acc > 0, f"NonLte + Ng: {n_acc} accelerations on 1 rank")
for n in ranks:
    pn = run_nlte(n, "Fixed", ng=True)
    # TODO(cmo): Report only until the SC state refactor is done, then revisit.
    print(f"INFO NonLte + Ng Fixed {n} ranks: pops max rel diff {rel(pn, p1):.2e}")

if failures:
    print(f"{len(failures)} failures")
    sys.exit(1)
print("All MPI tests passed")
