"""Refinement study of the vertical material interpolation on a non-uniform z
grid: FAL-C at its native heights (+ midpoints to 96 rows), refined by
repeated midpoint insertion (96, 192, 384 rows; the original nodes are kept),
for each interpolation. dx is small so characteristics cross several columns
per row (exercising the vertical interpolation at crossings). Plane parallel,
periodic x, thermalised bottom, zero top, H_6 non-LTE with fixed ne.
Usage: python sc_z_refinement.py DEXRT_SC WORKDIR [--rerun]"""
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
from lightweaver.fal import Falc82

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
nx, dx = 32, 10e3
levels = [1, 2, 4]
interps = ["Linear", "VanLeer", "MonotonizedCentral", "LongerCharacteristics"]

falc = Falc82()
up = lambda q: np.asarray(q)[::-1].astype(np.float64)
z0 = up(falc.height)
extra = (-len(z0)) % 32
split = np.sort(np.argsort(np.diff(z0))[-extra:])
z_base = np.sort(np.concatenate([z0, 0.5 * (z0[split] + z0[split + 1])]))


def refine(z, n):
    while n > 1:
        z = np.sort(np.concatenate([z, 0.5 * (z[1:] + z[:-1])]))
        z = np.concatenate([z, [z[-1] + 0.5 * (z[-1] - z[-2])]])  # keep a multiple of 32
        n //= 2
    return z


KB = 1.380649e-23


def write_atmos(level):
    # NOTE: Midpoint insertion gives 2n - 1 rows; one extra row above the top
    # keeps nz a multiple of 32 (the original nodes are unchanged).
    z = refine(z_base, level)
    lin = lambda q: np.interp(z, z0, up(q))
    log = lambda q: np.exp(np.interp(z, z0, np.log(up(q))))
    T, ne, nh, vturb = lin(falc.temperature), log(falc.ne), log(falc.nHTot), lin(falc.vturb)
    path = work / f"falc_x{level}.nc"
    with ncdf.Dataset(path, "w") as f:
        f.createDimension("x", nx)
        f.createDimension("z", z.shape[0])
        for k, v in dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=(1.1 * nh + ne) * KB * T,
                         vx=0 * T, vy=0 * T, vz=0 * T).items():
            f.createVariable(k, "f4", ("z", "x"))[...] = np.repeat(v[:, None], nx, axis=1)
        f.createVariable("voxel_scale", "f4")[...] = dx
        f.createVariable("z", "f8", ("z",))[...] = z
    return path, z


def run(level, interp, atmos_path):
    name = f"x{level}_{interp}"
    conf = {
        "mode": "NonLte",
        "atmos_path": str(atmos_path),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Thermalised",
        "conserve_charge": False,
        "pop_tol": 1e-5,
        "max_iter": 1000,
        "ng_config": {"enable": False},
        "system": {"mem_pool_gb": 3.0},
        "sc": {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
               "interpolation": interp, "x_boundary": "Periodic"},
    }
    if interp == "LongerCharacteristics":
        conf["sc"] = {"quadrature": "StepanMirrored", "quadrature_order": 7, "x_boundary": "Periodic",
                      "characteristics": "LongerCharacteristics"}
    # NOTE: Reuse a finished run with the same config (--rerun to force).
    conf_text = yaml.dump(conf)
    done = (work / f"{name}.nc").exists() and (work / f"{name}.yaml").exists() and (work / f"{name}.yaml").read_text() == conf_text
    if "--rerun" in sys.argv or not done:
        (work / f"{name}.yaml").write_text(conf_text)
        with open(work / f"{name}.log", "w") as log:
            subprocess.run([str(dexrt_sc), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    out = ncdf.Dataset(work / f"{name}.nc")
    nz = out["pops"].shape[1]
    return (np.asarray(out["pops"][...])[:, :nz, :nx].mean(axis=2),
            np.asarray(out["J"][...])[:, :nz, :nx].mean(axis=2))


res = {}
for level in levels:
    path, z = write_atmos(level)
    print(f"x{level}: {z.shape[0]} rows, dz {np.diff(z).min():.0f} .. {np.diff(z).max():.0f} m")
    for interp in interps:
        pops, J = run(level, interp, path)
        # NOTE: The base nodes sit at every level-th row.
        res[(level, interp)] = (pops[:, ::level][:, :z_base.shape[0]], J[:, ::level][:, :z_base.shape[0]])

ref_pops, ref_J = res[(levels[-1], "Linear")]


def err(a, b):
    r = np.abs(a - b) / np.maximum(np.abs(b), 1e-30)
    return f"median {np.median(r):.2e}, 90% {np.percentile(r, 90):.2e}, max {r.max():.2e}"


print(f"Errors at the 96 base nodes vs x{levels[-1]} Linear:")
for level in levels:
    for interp in interps:
        pops, J = res[(level, interp)]
        print(f"  x{level} {interp:20s} pops {err(pops, ref_pops)} | J {err(J, ref_J)}")
np.savez(work / "refinement.npz", z=z_base, **{f"pops_x{l}_{i}": res[(l, i)][0] for l in levels for i in interps},
         **{f"J_x{l}_{i}": res[(l, i)][1] for l in levels for i in interps})
