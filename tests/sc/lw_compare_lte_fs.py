"""Compare a single LTE formal solution (J) between dexrt_sc and Lightweaver
(2D, piecewise linear SC, linear interpolation, periodic x, zero z boundaries,
mirrored Stepan quadrature, no background opacity).
Usage: python lw_compare_lte_fs.py DEXRT_SC WORKDIR"""
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]

nx, nz = 64, 64
dx = 100e3
KB = 1.380649e-23

zz, xx = np.meshgrid((np.arange(nz) + 0.5) / nz, (np.arange(nx) + 0.5) / nx, indexing="ij")
T = 6000.0 + 4000.0 * np.exp(-((xx - 0.4) ** 2 + (zz - 0.55) ** 2) / 0.02) + 1500.0 * zz
nh = 3e16 * (1.0 + 2.0 * np.exp(-((xx - 0.6) ** 2 + (zz - 0.3) ** 2) / 0.03))
ne = 0.2 * nh
vturb = np.full_like(T, 5e3)


def run_dexrt(name, sc):
    atmos_path = work / f"{name}_atmos.nc"
    with ncdf.Dataset(atmos_path, "w") as f:
        f.createDimension("x", nx)
        f.createDimension("z", nz)
        for k, v in dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=nh * KB * T,
                         vx=0 * T, vy=0 * T, vz=0 * T).items():
            f.createVariable(k, "f4", ("z", "x"))[...] = v
        f.createVariable("voxel_scale", "f4")[...] = dx
    conf = {
        "mode": "Lte",
        "atmos_path": str(atmos_path),
        "output_path": str(work / f"{name}_out.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Zero",
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
    out = ncdf.Dataset(work / f"{name}_out.nc")
    return np.asarray(out["wavelength"][...]), np.asarray(out["J"][...])[:, :nz, :nx]


def run_lw(nrays):
    # NOTE: Lightweaver pins I to the boundary on its outermost rows, so add a
    # near-transparent pad row top and bottom (same T, so the same S) to
    # match the half-cell ghost treatment in dexrt_sc. Periodic x in
    # Lightweaver identifies the first and last columns: add one.
    def pad(a, scale=1.0):
        a = np.concatenate([a[:1] * scale, a, a[-1:] * scale], axis=0)
        return np.concatenate([a, a[:, :1]], axis=1)
    # Lightweaver is top-first
    flip = lambda a: np.ascontiguousarray(a[::-1].astype(np.float64))
    height = ((np.arange(nz + 2) - 0.5) * dx)[::-1].copy()
    x = (np.arange(nx + 1) + 0.5) * dx
    atmos = lw.Atmosphere.make_2d(
        height=height,
        x=x,
        temperature=flip(pad(T)),
        vx=flip(pad(0 * T)),
        vz=flip(pad(0 * T)),
        vturb=flip(pad(vturb)),
        ne=flip(pad(ne, 1e-12)),
        nh_tot=flip(pad(nh, 1e-12)),
        z_lower_bc=lw.ZeroRadiation(),
        z_upper_bc=lw.ZeroRadiation(),
    )
    atmos.quadrature(nrays)
    a_set = lw.RadiativeSet([H_6_atom()])
    a_set.set_detailed_static("H")
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16,
                     formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
    ctx.background.eta[...] = 0.0
    ctx.background.chi[...] = 0.0
    ctx.background.sca[...] = 0.0
    ctx.depthData.fill = True
    ctx.formal_sol_gamma_matrices()
    wl = np.asarray(spect.wavelength)
    J = np.asarray(ctx.spect.J).reshape(wl.shape[0], nz + 2, nx + 1)
    J = J[:, ::-1][:, 1:-1, :nx]
    to_dex = lambda la, a: lw.convert_specific_intensity(wl[la], a, outUnits="kW / (m2 nm sr)").value
    J = np.stack([to_dex(la, J[la]) for la in range(wl.shape[0])])
    # NOTE: No velocities, so ray 0 holds the material for every ray.
    shape = (wl.shape[0], nz + 2, nx + 1)
    chi = np.asarray(ctx.depthData.chi)[:, 0, 0].reshape(shape)[:, ::-1][:, 1:-1, :nx]
    eta = np.asarray(ctx.depthData.eta)[:, 0, 0].reshape(shape)[:, ::-1][:, 1:-1, :nx]
    S = np.stack([to_dex(la, eta[la] / chi[la]) for la in range(wl.shape[0])])
    return wl, J, chi, S


nrays = 3
wl_lw, J_lw, chi_lw, S_lw = run_lw(nrays)
wl_dex, J_dex = run_dexrt("lte_fs", {"quadrature": "StepanMirrored", "quadrature_order": 7,
                                       "formal_solver": "PiecewiseLinear", "interpolation": "Linear",
                                       "x_boundary": "Periodic"})

# Compare on wavelengths present in both grids
idx_lw = []
idx_dex = []
for i, w in enumerate(wl_dex):
    j = np.argmin(np.abs(wl_lw - w))
    if abs(wl_lw[j] - w) < 1e-6 * w:
        idx_lw.append(j)
        idx_dex.append(i)
print(f"{len(idx_dex)} common wavelengths of {len(wl_dex)} (dexrt), {len(wl_lw)} (lw)")
A = J_dex[idx_dex]
B = J_lw[idx_lw]
mask = B > 1e-10 * B.max()
rel = np.abs(A - B)[mask] / B[mask]
print(f"J rel diff: median {np.median(rel):.2e}, 99% {np.percentile(rel, 99):.2e}, max {rel.max():.2e}")
worst = np.unravel_index(np.argmax(np.where(mask, np.abs(A - B) / np.where(mask, B, 1), 0)), A.shape)
print("worst at (wave, z, x)", worst, "lambda", wl_dex[idx_dex[worst[0]]], A[worst], B[worst])
np.savez(work / "lte_fs_compare.npz", wl=wl_dex[idx_dex], J_dex=A, J_lw=B)

# Solver only: Lightweaver's own eta/chi through dexrt_sc in GivenFs mode.
sys.path.insert(0, str(Path(__file__).parent))
from sc_givenfs_tests_lib import run_givenfs
chi_g = np.moveaxis(chi_lw, 0, -1).astype(np.float32) * dx
eta_g = (np.moveaxis(S_lw, 0, -1) * np.moveaxis(chi_lw, 0, -1) * dx).astype(np.float32)
out = run_givenfs(dexrt_sc, work, "lw_givenfs", eta_g, chi_g,
                  {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
                   "interpolation": "Linear", "x_boundary": "Periodic"}, voxel_scale=1.0)
J_g = np.moveaxis(np.asarray(out["J"][...])[:, :nz, :nx], 0, 0)
mask = J_lw > 1e-25
rel = np.abs(J_g - J_lw)[mask] / J_lw[mask]
np.savez(work / "solver_only.npz", J_g=J_g, J_lw=J_lw, tau_cell=np.moveaxis(chi_g, -1, 0), wl=wl_lw)
print(f"Solver-only (LW eta/chi) J rel diff: median {np.median(rel):.2e}, 99% {np.percentile(rel, 99):.2e}, max {rel.max():.2e}")
worst = np.unravel_index(np.argmax(np.where(mask, np.abs(J_g - J_lw) / np.where(mask, J_lw, 1), 0)), J_lw.shape)
print("  worst at (wave, z, x)", worst, "lambda", wl_lw[worst[0]], J_g[worst], J_lw[worst])
