"""Non-LTE H_6 on FAL-C at its native (non-uniform) heights: dexrt_sc (z
variable) vs Lightweaver 2D on the same grid. Plane parallel (periodic x),
fixed ne, no background, thermalised lower and zero upper boundary. Midpoints are added to
the largest steps so nz is a multiple of 32 (the SC slab unit). --uniform:
the same model resampled to uniform steps (control), via voxel_scale_z.
Usage: python lw_compare_native_z.py DEXRT_SC WORKDIR [--uniform]"""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
import lightweaver as lw
from lightweaver.fal import Falc82

sys.path.insert(0, str(Path(__file__).parent))
from lw_models import crtaf_h6  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dexrt_py"))
from dexrt.background_tables import _disable_debye  # noqa: E402

# NOTE: DexRT's LTE populations have no Debye (pressure ionisation) term.
_disable_debye()

uniform = "--uniform" in sys.argv
lte = "--lte" in sys.argv
sys.argv = [a for a in sys.argv if a not in ("--uniform", "--lte")]
tag = "uniform" if uniform else "native"
dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]

falc = Falc82()
# NOTE: dexrt: increasing upwards; Lightweaver/RH: decreasing.
up = lambda q: np.asarray(q)[::-1].astype(np.float64)
z0 = up(falc.height)
extra = (-len(z0)) % 32
split = np.sort(np.argsort(np.diff(z0))[-extra:])
z = np.sort(np.concatenate([z0, 0.5 * (z0[split] + z0[split + 1])]))
if uniform:
    z = np.linspace(z[0], z[-1], z.shape[0])
lin = lambda q: np.interp(z, z0, up(q))
log = lambda q: np.exp(np.interp(z, z0, np.log(up(q))))
T, ne, nh, vturb = lin(falc.temperature), log(falc.ne), log(falc.nHTot), lin(falc.vturb)
nz, nx = z.shape[0], 32
dx = float(os.environ.get("SC_LW_DX", 20e3))
steps = np.diff(z)
print(f"FAL-C {tag}: {nz} rows, dz {steps.min():.0f} .. {steps.max():.0f} m, max adjacent ratio {np.max(np.maximum(steps[1:] / steps[:-1], steps[:-1] / steps[1:])):.2f}")

KB = 1.380649e-23
atmos_path = work / f"falc_{tag}.nc"
with ncdf.Dataset(atmos_path, "w") as f:
    f.createDimension("x", nx)
    f.createDimension("z", nz)
    for k, v in dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=(1.1 * nh + ne) * KB * T,
                     vx=0 * T, vy=0 * T, vz=0 * T).items():
        f.createVariable(k, "f4", ("z", "x"))[...] = np.repeat(v[:, None], nx, axis=1)
    f.createVariable("voxel_scale", "f4")[...] = dx
    if uniform:
        f.createVariable("voxel_scale_z", "f4")[...] = z[1] - z[0]
        f.createVariable("offset_z", "f4")[...] = z[0] - 0.5 * (z[1] - z[0])
    else:
        f.createVariable("z", "f8", ("z",))[...] = z


def run_dexrt(name, interp):
    conf = {
        "mode": "NonLte",
        "atmos_path": str(atmos_path),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Thermalised",
        "conserve_charge": False,
        "pop_tol": 1e-4,
        "max_iter": 500,
        # NOTE: Ng (on by default) stalls this case at a few % change.
        "ng_config": {"enable": False},
        "system": {"mem_pool_gb": 2.0},
        "sc": {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
               "interpolation": interp, "x_boundary": "Periodic"},
    }
    if interp == "LongerCharacteristics":
        conf["sc"] = {"quadrature": "StepanMirrored", "quadrature_order": 7, "x_boundary": "Periodic",
                      "characteristics": "LongerCharacteristics"}
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    log_path = work / f"{name}.log"
    with open(log_path, "w") as log:
        res = subprocess.run([str(dexrt_sc), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(log_path.read_text()[-3000:])
        raise RuntimeError("dexrt_sc failed")
    n_fs = log_path.read_text().count("==== FS")
    out = ncdf.Dataset(work / f"{name}.nc")
    J = np.asarray(out["J"][...])[:, :nz, :nx].mean(axis=2)
    I = np.asarray(out["I_emergent"][...]).mean(axis=2)
    mu = np.asarray(out["sc_mu"][...])
    return np.asarray(out["pops"][...])[:, :nz, :nx].mean(axis=2), n_fs, np.asarray(out["wavelength"][...]), J, I, mu


def run_lw():
    # NOTE: As lw_compare_nlte.py: a near-void ghost row at each end (mirrored
    # spacing) and a periodic column.
    zg = np.concatenate([[2 * z[0] - z[1]], z, [2 * z[-1] - z[-2]]])
    def pad(a, scale=1.0):
        a = np.concatenate([a[:1] * scale, a, a[-1:] * scale])
        return np.repeat(a[:, None], nx + 1, axis=1)
    flip = lambda a: np.ascontiguousarray(a[::-1])
    atmos = lw.Atmosphere.make_2d(
        height=flip(zg), x=(np.arange(nx + 1) + 0.5) * dx,
        temperature=flip(pad(T)), vx=flip(pad(0 * T)), vz=flip(pad(0 * T)), vturb=flip(pad(vturb)),
        ne=flip(pad(ne, 1e-3)), nh_tot=flip(pad(nh, 1e-3)),
        z_lower_bc=lw.ThermalisedRadiation(), z_upper_bc=lw.ZeroRadiation(),
    )
    atmos.quadrature(3)
    a_set = lw.RadiativeSet([crtaf_h6()])
    a_set.set_active("H")
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16,
                     formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
    ctx.background.eta[...] = 0.0
    ctx.background.chi[...] = 0.0
    ctx.background.sca[...] = 0.0
    lw.iterate_ctx_se(ctx, pops_tol=1e-4, max_iter=500, quiet=True)
    wl = np.asarray(spect.wavelength)
    J = np.asarray(ctx.spect.J).reshape(wl.shape[0], nz + 2, nx + 1)[:, ::-1][:, 1:-1, :nx].mean(axis=2)
    J = np.stack([lw.convert_specific_intensity(wl[la], J[la], outUnits="kW / (m2 nm sr)").value for la in range(wl.shape[0])])
    pops = np.asarray(eq_pops["H"]).reshape(-1, nz + 2, nx + 1)[:, ::-1][:, 1:-1, :nx].mean(axis=2)
    ctx.formal_sol_gamma_matrices()
    I = np.asarray(ctx.spect.I)
    mu = np.stack([np.asarray(atmos.muz), np.asarray(atmos.mux), np.asarray(atmos.wmu)], axis=-1)
    return pops, wl, J, I, mu


def run_dexrt_lte_J(name):
    conf = {
        "mode": "Lte",
        "atmos_path": str(atmos_path),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Thermalised",
        "system": {"mem_pool_gb": 2.0},
        "sc": {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
               "interpolation": "Linear", "x_boundary": "Periodic"},
    }
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    with open(work / f"{name}.log", "w") as log:
        subprocess.run([str(dexrt_sc), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    out = ncdf.Dataset(work / f"{name}.nc")
    return np.asarray(out["wavelength"][...]), np.asarray(out["J"][...])[:, :nz, :nx].mean(axis=2)


def run_lw_lte_J():
    zg = np.concatenate([[2 * z[0] - z[1]], z, [2 * z[-1] - z[-2]]])
    def pad(a, scale=1.0):
        a = np.concatenate([a[:1] * scale, a, a[-1:] * scale])
        return np.repeat(a[:, None], nx + 1, axis=1)
    flip = lambda a: np.ascontiguousarray(a[::-1])
    atmos = lw.Atmosphere.make_2d(
        height=flip(zg), x=(np.arange(nx + 1) + 0.5) * dx,
        temperature=flip(pad(T)), vx=flip(pad(0 * T)), vz=flip(pad(0 * T)), vturb=flip(pad(vturb)),
        ne=flip(pad(ne, 1e-6)), nh_tot=flip(pad(nh, 1e-6)),
        z_lower_bc=lw.ThermalisedRadiation(), z_upper_bc=lw.ZeroRadiation(),
    )
    atmos.quadrature(3)
    a_set = lw.RadiativeSet([crtaf_h6()])
    a_set.set_detailed_static("H")
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16,
                     formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
    ctx.background.eta[...] = 0.0
    ctx.background.chi[...] = 0.0
    ctx.background.sca[...] = 0.0
    ctx.formal_sol_gamma_matrices()
    wl = np.asarray(spect.wavelength)
    J = np.asarray(ctx.spect.J).reshape(wl.shape[0], nz + 2, nx + 1)[:, ::-1][:, 1:-1, :nx].mean(axis=2)
    J = np.stack([lw.convert_specific_intensity(wl[la], J[la], outUnits="kW / (m2 nm sr)").value for la in range(wl.shape[0])])
    return wl, J


if lte:
    wl_d, J_d = run_dexrt_lte_J(f"{tag}_lte")
    wl_l, J_l = run_lw_lte_J()
    pairs = [(i, np.argmin(np.abs(wl_l - w))) for i, w in enumerate(wl_d)]
    pairs = [(i, j) for i, j in pairs if abs(wl_l[j] - wl_d[i]) < 1e-6 * wl_d[i]]
    A = J_d[[i for i, _ in pairs]]
    B = J_l[[j for _, j in pairs]]
    r = np.abs(A - B) / np.maximum(B, 1e-30)
    print(f"LTE J, {len(pairs)} common wavelengths: rel diff median {np.median(r):.2e}, 90% {np.percentile(r, 90):.2e}, 99% {np.percentile(r, 99):.2e}")
    for k in range(0, nz, 6):
        print(f"  row {k:3d} z = {z[k] / 1e3:8.1f} km T = {T[k]:7.0f}: median {np.median(r[:, k]):.2e}, max {r[:, k].max():.2e}")
    np.savez(work / f"{tag}_lte_J.npz", wl=wl_d[[i for i, _ in pairs]], J_dex=A, J_lw=B, z=z)
    sys.exit(0)

pops_lw, wl_lw, J_lw, I_lw, mu_lw = run_lw()
np.savez(work / f"{tag}_lw.npz", pops=pops_lw, z=z, wl=wl_lw, J=J_lw, I=I_lw, mu=mu_lw)
for interp in ["Linear", "VanLeer", "MonotonizedCentral", "LongerCharacteristics"]:
    pops_dex, n_fs, wl_dex, J_dex, I_dex, mu_dex = run_dexrt(f"{tag}_{interp}", interp)
    print(f"-- dexrt_sc {interp}: {n_fs} formal solutions")
    for lvl in range(pops_lw.shape[0]):
        rel = np.abs(pops_dex[lvl] - pops_lw[lvl]) / pops_lw[lvl]
        worst = np.argmax(rel)
        print(f"  level {lvl}: median {np.median(rel):.2e}, 90% {np.percentile(rel, 90):.2e}, max {rel.max():.2e} (z = {z[worst] / 1e3:.0f} km)")
    np.savez(work / f"{tag}_{interp}.npz", pops=pops_dex, z=z, wl=wl_dex, J=J_dex, I=I_dex, mu=mu_dex)
