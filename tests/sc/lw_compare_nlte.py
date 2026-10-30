"""Non-LTE H_6 comparison between dexrt_sc and Lightweaver on the same 2D
atmosphere as lw_compare_lte_fs.py (fixed ne, no background, periodic x).
Usage: python lw_compare_nlte.py DEXRT_SC WORKDIR"""
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom

sys.path.insert(0, str(Path(__file__).parent))
import lw_compare_lte_fs as base  # noqa: E402  (runs the LTE comparison too)

dexrt_sc, work, repo = base.dexrt_sc, base.work, base.repo
nx, nz, dx, T, nh, ne, vturb = base.nx, base.nz, base.dx, base.T, base.nh, base.ne, base.vturb


def run_dexrt_nlte(name, sc):
    conf = {
        "mode": "NonLte",
        "atmos_path": str(work / "lte_fs_atmos.nc"),
        "output_path": str(work / f"{name}_out.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Zero",
        "conserve_charge": False,
        "pop_tol": 1e-4,
        "max_iter": 500,
        "system": {"mem_pool_gb": 2.0},
        "sc": sc,
    }
    conf_path = work / f"{name}.yaml"
    conf_path.write_text(yaml.dump(conf))
    log_path = work / f"{name}.log"
    with open(log_path, "w") as log:
        res = subprocess.run([str(dexrt_sc), "--config", str(conf_path)], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(log_path.read_text()[-3000:])
        raise RuntimeError("dexrt_sc failed")
    n_fs = log_path.read_text().count("==== FS")
    out = ncdf.Dataset(work / f"{name}_out.nc")
    return np.asarray(out["pops"][...])[:, :nz, :nx], n_fs


def run_lw_nlte(nrays):
    def pad(a, scale=1.0):
        a = np.concatenate([a[:1] * scale, a, a[-1:] * scale], axis=0)
        return np.concatenate([a, a[:, :1]], axis=1)
    flip = lambda a: np.ascontiguousarray(a[::-1].astype(np.float64))
    height = ((np.arange(nz + 2) - 0.5) * dx)[::-1].copy()
    x = (np.arange(nx + 1) + 0.5) * dx
    atmos = lw.Atmosphere.make_2d(
        height=height, x=x,
        temperature=flip(pad(T)), vx=flip(pad(0 * T)), vz=flip(pad(0 * T)), vturb=flip(pad(vturb)),
        ne=flip(pad(ne, 1e-12)), nh_tot=flip(pad(nh, 1e-12)),
        z_lower_bc=lw.ZeroRadiation(), z_upper_bc=lw.ZeroRadiation(),
    )
    atmos.quadrature(nrays)
    a_set = lw.RadiativeSet([H_6_atom()])
    a_set.set_active("H")
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16,
                     formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
    ctx.background.eta[...] = 0.0
    ctx.background.chi[...] = 0.0
    ctx.background.sca[...] = 0.0
    lw.iterate_ctx_se(ctx, pops_tol=1e-4, max_iter=500, quiet=True)
    pops = np.asarray(eq_pops["H"]).reshape(-1, nz + 2, nx + 1)[:, ::-1][:, 1:-1, :nx]
    return pops


pops_lw = run_lw_nlte(3)
for interp in ["Linear", "MonotonizedCentral"]:
    pops_dex, n_fs = run_dexrt_nlte(f"nlte_{interp}", {
        "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
        "interpolation": interp, "x_boundary": "Periodic"})
    print(f"-- dexrt_sc {interp}: {n_fs} formal solutions")
    for lvl in range(pops_lw.shape[0]):
        rel = np.abs(pops_dex[lvl] - pops_lw[lvl]) / pops_lw[lvl]
        print(f"  level {lvl}: median {np.median(rel):.2e}, max {rel.max():.2e}")
    np.savez(work / f"nlte_compare_{interp}.npz", pops_dex=pops_dex, pops_lw=pops_lw)
