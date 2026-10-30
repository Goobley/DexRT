"""PRD (Hybrid) in moving 3D media, H_6 on FAL-C (native heights, periodic
x/y, thermalised bottom), dexrt_sc_3d.
 1. A vertical shear of a horizontal velocity, v_x(z): 3D against 2D, and 3D
    with the same shear in v_y (the Stepan set is invariant under rotations
    of pi/2, so all three agree to convergence level).
 2. A genuinely 3D shear, v_x(y) and v_y(x): converges; 1 vs 2 ranks (CPU,
    fixed iterations) agree to rounding.
Usage: python sc_3d_prd.py DEXRT_SC DEXRT_SC_3D DEXRT_SC_3D_CPU_MPI WORKDIR [Short|LongerCharacteristics]"""
import os
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
from lightweaver.fal import Falc82

dexrt_sc, dexrt_sc_3d, dexrt_sc_3d_cpu, work = (Path(a).resolve() for a in sys.argv[1:5])
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
mpi = shlex.split(os.environ.get(
    "SC_MPIEXEC",
    "mpirun --oversubscribe --bind-to none --mca pml ob1 --mca btl self,tcp --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0",
))

falc = Falc82()
up = lambda q: np.asarray(q)[::-1].astype(np.float64)
z0 = up(falc.height)
extra = (-len(z0)) % 16
split = np.sort(np.argsort(np.diff(z0))[-extra:])
z = np.sort(np.concatenate([z0, 0.5 * (z0[split] + z0[split + 1])]))
lin = lambda q: np.interp(z, z0, up(q))
log = lambda q: np.exp(np.interp(z, z0, np.log(up(q))))
T, ne, nh, vturb = lin(falc.temperature), log(falc.ne), log(falc.nh_tot), lin(falc.vturb)
nz, n = z.shape[0], 16
KB = 1.380649e-23
shear = 20e3 * np.tanh((z - 1.5e6) / 3e5)


def write(path, three_d, vx, vy):
    """vx, vy: [z] or [z, y, x] (3D)."""
    with ncdf.Dataset(path, "w") as f:
        f.createDimension("x", n)
        if three_d:
            f.createDimension("y", n)
        f.createDimension("z", nz)
        fields = dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=(1.1 * nh + ne) * KB * T, vz=0 * T)
        for k, v in fields.items():
            if three_d:
                f.createVariable(k, "f4", ("z", "y", "x"))[...] = np.broadcast_to(v[:, None, None], (nz, n, n))
            else:
                f.createVariable(k, "f4", ("z", "x"))[...] = np.repeat(v[:, None], n, axis=1)
        for k, v in (("vx", vx), ("vy", vy)):
            if three_d:
                f.createVariable(k, "f4", ("z", "y", "x"))[...] = np.broadcast_to(v if v.ndim == 3 else v[:, None, None], (nz, n, n))
            else:
                f.createVariable(k, "f4", ("z", "x"))[...] = np.repeat(v[:, None], n, axis=1)
        f.createVariable("voxel_scale", "f4")[...] = 20e3
        f.createVariable("z", "f8", ("z",))[...] = z


def run(binary, name, atmos, three_d, characteristics, max_iter=500, pop_tol=1e-4, ranks=0):
    sc = {"quadrature": "Stepan", "quadrature_order": 7, "formal_solver": "PiecewiseLinear", "interpolation": "Linear",
          "x_boundary": "Periodic", "characteristics": characteristics, "prd": "Hybrid"}
    if three_d:
        sc["y_boundary"] = "Periodic"
    conf = {
        "mode": "NonLte",
        "atmos_path": str(atmos),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Thermalised",
        "conserve_charge": False,
        "pop_tol": pop_tol,
        "max_iter": max_iter,
        "ng_config": {"enable": False},
        "system": {"mem_pool_gb": 2.0},
        "sc": sc,
    }
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    cmd = [str(binary), "--config", str(work / f"{name}.yaml")]
    env = dict(os.environ)
    if ranks:
        cmd = mpi + ["-n", str(ranks)] + cmd
        env.update(OMP_NUM_THREADS="4", OMP_PROC_BIND="false")
    logf = work / f"{name}.log"
    with open(logf, "w") as f:
        r = subprocess.run(cmd, cwd=work, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        print(logf.read_text()[-3000:])
        raise RuntimeError(f"{name} failed")
    out = ncdf.Dataset(work / f"{name}.nc")
    return np.asarray(out["pops"][...]), np.asarray(out["prd_rho"][...]), logf.read_text().count("==== FS")


zero = 0 * T
write(work / "vx_2d.nc", False, shear, zero)
write(work / "vx_3d.nc", True, shear, zero)
write(work / "vy_3d.nc", True, zero, shear)
yy, xx = np.mgrid[0:n, 0:n] / n
amp = np.where(z > 1.0e6, 15e3, 0.0)[:, None, None]
vx3 = amp * np.sin(2 * np.pi * yy)[None]
vy3 = amp * np.sin(2 * np.pi * xx)[None]
write(work / "shear_3d.nc", True, vx3, vy3)

only = sys.argv[5:]
# NOTE: 3D vs 2D: short characteristics also differ by truncation (the y
# crossings split the formal steps; ~7% max at FAL-C's native steps, see
# sc_3d_falc.py); the vy vs vx rotation is exact for both.
for ch, tol, tol_rot in (("LongerCharacteristics", 5e-3, 5e-3), ("Short", 0.1, 1e-4)):
    if only and ch not in only:
        continue
    p2, r2, n2 = run(dexrt_sc, f"{ch}_vx_2d", work / "vx_2d.nc", False, ch)
    p3, r3, n3 = run(dexrt_sc_3d, f"{ch}_vx_3d", work / "vx_3d.nc", True, ch)
    p3y, r3y, n3y = run(dexrt_sc_3d, f"{ch}_vy_3d", work / "vy_3d.nc", True, ch)
    col2, col3, col3y = p2.mean(axis=2), p3.mean(axis=(2, 3)), p3y.mean(axis=(2, 3))
    d3 = np.abs(col3 - col2) / col2
    dy = np.abs(col3y - col3) / col3
    print(f"{ch} v(z) shear: iterations 2D {n2}, 3D vx {n3}, 3D vy {n3y}; "
          f"|3D vx - 2D| median {np.median(d3):.1e} max {np.max(d3):.1e}; "
          f"|3D vy - 3D vx| median {np.median(dy):.1e} max {np.max(dy):.1e}; rho in [{r3.min():.2f}, {r3.max():.2f}]")
    assert np.max(d3) < tol and np.max(dy) < tol_rot, (np.max(d3), np.max(dy))
    p, r, nit = run(dexrt_sc_3d, f"{ch}_shear_3d", work / "shear_3d.nc", True, ch)
    lvl = p.mean(axis=(2, 3))
    horiz = np.max(np.abs(p - lvl[:, :, None, None]) / lvl[:, :, None, None])
    print(f"{ch} 3D shear: {nit} iterations, horizontal pops variation {horiz:.2e}, rho in [{r.min():.2f}, {r.max():.2f}]")
    assert nit < 500
    a, ra, _ = run(dexrt_sc_3d_cpu, f"{ch}_shear_mpi1", work / "shear_3d.nc", True, ch, max_iter=6, pop_tol=1e-12, ranks=1)
    b, rb, _ = run(dexrt_sc_3d_cpu, f"{ch}_shear_mpi2", work / "shear_3d.nc", True, ch, max_iter=6, pop_tol=1e-12, ranks=2)
    dm = np.max(np.abs(b - a) / a)
    dr = np.max(np.abs(rb - ra))
    print(f"{ch} 3D shear, 2 ranks vs 1 (6 iterations): max rel pops {dm:.1e}, max |drho| {dr:.1e}")
    assert dm < 1e-5 and dr < 1e-5, (dm, dr)
print("3D PRD: passed")
