"""Non-LTE H_6 with Lyman alpha/beta in PRD on FAL-C at its native heights:
dexrt_sc (sc.prd: Hybrid) vs Lightweaver 2D (prd, hybrid if moving), and both
in CRD for scale. Plane parallel (periodic x), fixed ne, no background,
thermalised lower and zero upper boundary, as lw_compare_native_z.py.
--vz V: a constant vertical velocity V km/s. --shear S: plus S sin(2 pi x / L)
km/s (a periodic shear in v_z, the angle-dependent test of the hybrid
transforms). Populations are compared cell by cell, and the emergent line
profiles (mean over x) for each upgoing direction both codes share: the
populations barely depend on the sign of the Doppler shift, the profiles do.
Usage: python lw_compare_prd.py DEXRT_SC WORKDIR [--vz V] [--shear S]"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
import lightweaver as lw
import lightweaver.atomic_model as lw_am
import lightweaver.collisional_rates as lw_cr
from lightweaver.fal import Falc82

sys.path.insert(0, str(Path(__file__).parent))
from lw_models import crtaf_h6  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dexrt_py"))
from dexrt.background_tables import _disable_debye  # noqa: E402

_disable_debye()
# NOTE: DexRT interpolates collision rates and cross sections linearly.
_linear = lambda x, xp, fp, left=None, right=None: np.interp(x, xp, fp, left=left, right=right)
lw_cr.weno4 = _linear
lw_am.weno4 = _linear

parser = argparse.ArgumentParser()
parser.add_argument("dexrt_sc")
parser.add_argument("work")
parser.add_argument("--vz", type=float, default=0.0)
parser.add_argument("--shear", type=float, default=0.0)
parser.add_argument("--prd-tol", type=float, default=1e-2, help="sub-iteration tolerance (both codes)")
parser.add_argument("--pop-tol", type=float, default=1e-4)
parser.add_argument("--max-iter", type=int, default=500)
parser.add_argument("--skip-lw", action="store_true")
parser.add_argument("--skip-dexrt", action="store_true")
parser.add_argument("--prd-only", action="store_true", help="skip the CRD runs")
parser.add_argument("--crd-only", action="store_true", help="skip the PRD runs")
args = parser.parse_args()
dexrt_sc = Path(args.dexrt_sc).resolve()
work = Path(args.work).resolve()
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
tag = f"vz{args.vz:+.0f}_shear{args.shear:.0f}"
# NOTE: An MPI build needs a launcher here (SC_LAUNCH="" for a non-MPI build).
launch = shlex.split(os.environ.get("SC_LAUNCH", "mpirun -n 1 --mca pml ob1 --mca btl self,tcp --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0"))
# NOTE: Lightweaver identifies its column nx with column 0, so column i of
# both codes is at x = i dx.
nx = 32
vz_x = (args.vz + args.shear * np.sin(2 * np.pi * np.arange(nx + 1) / nx)) * 1e3
moving = np.any(vz_x != 0.0)

falc = Falc82()
up = lambda q: np.asarray(q)[::-1].astype(np.float64)
z0 = up(falc.height)
extra = (-len(z0)) % 32
split = np.sort(np.argsort(np.diff(z0))[-extra:])
z = np.sort(np.concatenate([z0, 0.5 * (z0[split] + z0[split + 1])]))
lin = lambda q: np.interp(z, z0, up(q))
log = lambda q: np.exp(np.interp(z, z0, np.log(up(q))))
T, ne, nh, vturb = lin(falc.temperature), log(falc.ne), log(falc.nHTot), lin(falc.vturb)
nz, dx = z.shape[0], 20e3

KB = 1.380649e-23
atmos_path = work / f"falc_{tag}.nc"
with ncdf.Dataset(atmos_path, "w") as f:
    f.createDimension("x", nx)
    f.createDimension("z", nz)
    for k, v in dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=(1.1 * nh + ne) * KB * T,
                     vx=0 * T, vy=0 * T).items():
        f.createVariable(k, "f4", ("z", "x"))[...] = np.repeat(v[:, None], nx, axis=1)
    f.createVariable("vz", "f4", ("z", "x"))[...] = np.repeat(vz_x[None, :nx], nz, axis=0)
    f.createVariable("voxel_scale", "f4")[...] = dx
    f.createVariable("z", "f8", ("z",))[...] = z


def run_dexrt(name, prd):
    conf = {
        "mode": "NonLte",
        "atmos_path": str(atmos_path),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Thermalised",
        "conserve_charge": False,
        "pop_tol": args.pop_tol,
        "max_iter": args.max_iter,
        "ng_config": {"enable": False},
        "system": {"mem_pool_gb": 2.0},
        "sc": {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
               "interpolation": "Linear", "x_boundary": "Periodic", "prd": prd,
               "prd_subiterations": 3, "prd_tolerance": args.prd_tol},
    }
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    log_path = work / f"{name}.log"
    with open(log_path, "w") as log:
        res = subprocess.run(launch + [str(dexrt_sc), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(log_path.read_text()[-3000:])
        raise RuntimeError("dexrt_sc failed")
    n_fs = log_path.read_text().count("==== FS")
    out = ncdf.Dataset(work / f"{name}.nc")
    pops = np.asarray(out["pops"][...])[:, :nz, :nx]
    rho = {}
    if "prd_rho" in out.variables:
        table = np.asarray(out["prd_lines"][...])
        rho_all = np.asarray(out["prd_rho"][...])[:, :nz, :nx]
        for kr, start, count, _ in table:
            rho[int(kr)] = rho_all[start:start + count]
    return dict(pops=pops, n_fs=n_fs, rho=rho, wl=np.asarray(out["wavelength"][...]),
                I=np.asarray(out["I_emergent"][...]).mean(axis=2), mu=np.asarray(out["sc_mu"][...]))


def run_lw(prd):
    zg = np.concatenate([[2 * z[0] - z[1]], z, [2 * z[-1] - z[-2]]])
    def pad(a, scale=1.0):
        a = np.concatenate([a[:1] * scale, a, a[-1:] * scale])
        return np.repeat(a[:, None], nx + 1, axis=1)
    flip = lambda a: np.ascontiguousarray(a[::-1])
    # NOTE: vz is positive upwards in both (Lightweaver: vlos = muz vz towards
    # the observer); only the array order is flipped.
    vz_lw = np.repeat(vz_x[None, :], nz + 2, axis=0)
    atmos = lw.Atmosphere.make_2d(
        height=flip(zg), x=np.arange(nx + 1) * dx,
        temperature=flip(pad(T)), vx=flip(pad(0 * T)), vz=flip(vz_lw), vturb=flip(pad(vturb)),
        ne=flip(pad(ne, 1e-3)), nh_tot=flip(pad(nh, 1e-3)),
        z_lower_bc=lw.ThermalisedRadiation(), z_upper_bc=lw.ZeroRadiation(),
    )
    atmos.quadrature(3)
    atom = crtaf_h6()
    if not prd:
        for l in atom.lines:
            l.type = lw.atomic_model.LineType.CRD
    a_set = lw.RadiativeSet([atom])
    a_set.set_active("H")
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16, hprd=prd and moving,
                     formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
    ctx.background.eta[...] = 0.0
    ctx.background.chi[...] = 0.0
    ctx.background.sca[...] = 0.0
    it = lw.iterate_ctx_se(ctx, print_interval=30.0, pops_tol=args.pop_tol, max_iter=args.max_iter, quiet=False, prd=prd, prd_iter_tol=args.prd_tol, max_prd_sub_iter=3)
    reshape = lambda a: np.asarray(a).reshape(a.shape[0], nz + 2, nx + 1)[:, ::-1][:, 1:-1, :nx]
    pops = reshape(np.asarray(eq_pops["H"]))
    rho = {}
    # NOTE: The lines come first in trans, in the atom's order (as dexrt's kr).
    for kr, t in enumerate(ctx.activeAtoms[0].trans):
        r = getattr(t, "rhoPrd", None)
        if r is not None and np.asarray(r).size > 0:
            rho[kr] = reshape(np.asarray(r))
    ctx.formal_sol_gamma_matrices()
    wl = np.asarray(spect.wavelength)
    I = np.asarray(ctx.spect.I)
    I = np.stack([lw.convert_specific_intensity(wl[la], I[la], outUnits="kW / (m2 nm sr)").value for la in range(wl.shape[0])])
    mu = np.stack([np.asarray(atmos.muz), np.asarray(atmos.mux), np.asarray(atmos.wmu)], axis=-1)
    return dict(pops=pops, n_fs=it, rho=rho, wl=wl, I=I, mu=mu)


def compare_pops(name, a, b):
    print(f"-- {name}")
    for lvl in range(b["pops"].shape[0]):
        rel = np.abs(a["pops"][lvl] - b["pops"][lvl]) / b["pops"][lvl]
        worst = np.unravel_index(np.argmax(rel), rel.shape)
        print(f"  level {lvl}: median {np.median(rel):.2e}, 90% {np.percentile(rel, 90):.2e}, max {rel.max():.2e} (z = {z[worst[0]] / 1e3:.0f} km, x = {worst[1]})")


def compare_emergent(name, a, b):
    """a: dexrt (I [wl, dir], mu [dir, (x, y, z)]), b: Lightweaver (I [wl, mu,
    x], mu [mu, (z, x, w)]). Lines' windows of +-1e-3 lambda0 (300 km/s)."""
    print(f"-- {name}: emergent line profiles, max |dI| / I")
    lam0 = [l.lambda0 for l in crtaf_h6().lines]
    for j in range(b["mu"].shape[0]):
        muz, mux = b["mu"][j, 0], b["mu"][j, 1]
        match = np.flatnonzero((np.abs(a["mu"][:, 2] - muz) < 1e-5) & (np.abs(a["mu"][:, 0] - mux) < 1e-5))
        if match.size == 0:
            continue
        I_a = a["I"][:, match[0]]
        I_b = np.interp(a["wl"], b["wl"], b["I"][:, j, :nx].mean(axis=1))
        worst = []
        for l0 in lam0:
            m = np.abs(a["wl"] - l0) < 1e-3 * l0
            worst.append(np.max(np.abs(I_a[m] - I_b[m]) / I_b[m]))
        print(f"  mu_z {muz:.3f}, mu_x {mux:.3f}: " + ", ".join(f"{l0:.1f} nm {w:.1e}" for l0, w in zip(lam0, worst)))


results = {}
for prd in [True] if args.prd_only else ([False] if args.crd_only else [True, False]):
    if not args.skip_dexrt:
        results[f"dex_{prd}"] = run_dexrt(f"{tag}_{'prd' if prd else 'crd'}", "Hybrid" if prd else "Off")
        print(f"dexrt_sc prd={prd}: {results[f'dex_{prd}']['n_fs']} formal solutions", flush=True)
    if not args.skip_lw:
        results[f"lw_{prd}"] = run_lw(prd)
        print(f"Lightweaver prd={prd}: {results[f'lw_{prd}']['n_fs']} iterations", flush=True)
np.savez(work / f"{tag}.npz", **{f"{k}_{q}": v[q] for k, v in results.items() for q in ("pops", "wl", "I", "mu")},
         **{f"{k}_rho_{kr}": r for k, v in results.items() for kr, r in v["rho"].items()}, z=z)
if "dex_False" in results and "dex_True" in results:
    compare_pops("dexrt PRD vs CRD (size of the PRD effect)", results["dex_True"], results["dex_False"])
if "lw_False" in results and "dex_False" in results:
    compare_pops("dexrt CRD vs Lightweaver CRD", results["dex_False"], results["lw_False"])
    compare_emergent("dexrt CRD vs Lightweaver CRD", results["dex_False"], results["lw_False"])
if "lw_True" in results and "dex_True" in results:
    compare_pops("dexrt PRD vs Lightweaver PRD", results["dex_True"], results["lw_True"])
    compare_emergent("dexrt PRD vs Lightweaver PRD", results["dex_True"], results["lw_True"])
    for kr, r_lw in results["lw_True"]["rho"].items():
        r_dex = results["dex_True"]["rho"].get(kr)
        if r_dex is None or r_dex.shape != r_lw.shape:
            print(f"  rho line {kr}: shapes {None if r_dex is None else r_dex.shape} vs {r_lw.shape}")
            continue
        d = np.abs(r_dex - r_lw)
        print(f"  rho line {kr}: |drho| median {np.median(d):.2e}, 99% {np.percentile(d, 99):.2e}, max {d.max():.2e}; rho range LW [{r_lw.min():.3f}, {r_lw.max():.3f}]")
