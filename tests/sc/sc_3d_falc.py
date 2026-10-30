"""Non-LTE H_6 on a horizontally homogeneous FAL-C (native heights, periodic
x and y, thermalised lower boundary): dexrt_sc_3d against the 2D dexrt_sc
(itself compared with Lightweaver in lw_compare_native_z.py), for short and
longer characteristics.
Usage: python sc_3d_falc.py DEXRT_SC DEXRT_SC_3D WORKDIR [--refine N]
(--refine: N - 1 extra heights in every step, for the short characteristics
truncation study)."""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
from lightweaver.fal import Falc82

refine = int(sys.argv[sys.argv.index("--refine") + 1]) if "--refine" in sys.argv else 1
dexrt_sc, dexrt_sc_3d, work = (Path(a).resolve() for a in sys.argv[1:4])
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]

falc = Falc82()
up = lambda q: np.asarray(q)[::-1].astype(np.float64)
z0 = up(falc.height)
# NOTE: nz a multiple of the 3D tile (8) and the 2D tile (16).
extra = (-len(z0)) % 16
split = np.sort(np.argsort(np.diff(z0))[-extra:])
z = np.sort(np.concatenate([z0, 0.5 * (z0[split] + z0[split + 1])]))
if refine > 1:
    z = np.interp(np.arange((len(z) - 1) * refine + 1) / refine, np.arange(len(z)), z)
    z = z[: len(z) - len(z) % 16]
lin = lambda q: np.interp(z, z0, up(q))
log = lambda q: np.exp(np.interp(z, z0, np.log(up(q))))
T, ne, nh, vturb = lin(falc.temperature), log(falc.ne), log(falc.nHTot), lin(falc.vturb)
nz, n = z.shape[0], 16
KB = 1.380649e-23
fields = dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=(1.1 * nh + ne) * KB * T, vx=0 * T, vy=0 * T, vz=0 * T)


def write(path, three_d):
    with ncdf.Dataset(path, "w") as f:
        f.createDimension("x", n)
        if three_d:
            f.createDimension("y", n)
        f.createDimension("z", nz)
        for k, v in fields.items():
            if three_d:
                f.createVariable(k, "f4", ("z", "y", "x"))[...] = np.broadcast_to(v[:, None, None], (nz, n, n))
            else:
                f.createVariable(k, "f4", ("z", "x"))[...] = np.repeat(v[:, None], n, axis=1)
        f.createVariable("voxel_scale", "f4")[...] = 20e3
        f.createVariable("z", "f8", ("z",))[...] = z


def run(binary, name, three_d, characteristics, mode="NonLte"):
    atmos = work / f"falc_{'3d' if three_d else '2d'}.nc"
    write(atmos, three_d)
    sc = {"quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
          "interpolation": "Linear", "x_boundary": "Periodic", "characteristics": characteristics}
    if three_d:
        sc["y_boundary"] = "Periodic"
    conf = {
        "mode": mode,
        "atmos_path": str(atmos),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Thermalised",
        "conserve_charge": False,
        "pop_tol": 1e-4,
        "max_iter": 500,
        "ng_config": {"enable": False},
        "system": {"mem_pool_gb": 2.0},
        "sc": sc,
    }
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    log = work / f"{name}.log"
    with open(log, "w") as f:
        r = subprocess.run([str(binary), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(log.read_text()[-3000:])
        raise RuntimeError(f"{name} failed")
    out = ncdf.Dataset(work / f"{name}.nc")
    if mode == "Lte":
        return np.asarray(out["J"][...]), 0
    pops = np.asarray(out["pops"][...])
    n_fs = log.read_text().count("==== FS")
    return pops, n_fs


# NOTE: LTE: one formal solution, J agrees to rounding for longer
# characteristics (truncation for short).
for ch, tol in (("LongerCharacteristics", 1e-5), ("Short", 1e-2)):
    J2, _ = run(dexrt_sc, f"{ch}_2d_lte", False, ch, "Lte")
    J3, _ = run(dexrt_sc_3d, f"{ch}_3d_lte", True, ch, "Lte")
    J2, J3 = J2.mean(axis=2), J3.mean(axis=(2, 3))
    scale = np.max(np.abs(J2), axis=1, keepdims=True)
    rel = (np.abs(J3 - J2) / np.where(scale > 0, scale, 1.0))[J2 > 1e-6 * scale]
    print(f"{ch} LTE J: |3D - 2D| / (max over z) median {np.median(rel):.2e} max {np.max(rel):.2e}")
    assert np.max(rel) < tol, np.max(rel)

# NOTE: non-LTE: each converged to pop_tol 1e-4 (f32), so longer agrees to the
# convergence error; short also differs by truncation (the 3D y crossings
# split the formal steps; ~7% max at the native steps, 0.6% with --refine 2).
for ch, tol in (("LongerCharacteristics", 2e-3), ("Short", 0.1 if refine == 1 else 2e-2)):
    p2, n2 = run(dexrt_sc, f"{ch}_2d", False, ch)
    p3, n3 = run(dexrt_sc_3d, f"{ch}_3d", True, ch)
    col2 = p2.mean(axis=2)
    col3 = p3.mean(axis=(2, 3))
    horiz = np.max(np.abs(p3 - col3[:, :, None, None]) / col3[:, :, None, None])
    rel = np.abs(col3 - col2) / col2
    print(f"{ch}: iterations 2D {n2}, 3D {n3}; 3D horizontal variation {horiz:.1e}; "
          f"pops |3D - 2D| / 2D median {np.median(rel):.2e} max {np.max(rel):.2e}")
    assert horiz < 1e-5, horiz
    assert np.max(rel) < tol, np.max(rel)
print("FAL-C 3D vs 2D: passed")
