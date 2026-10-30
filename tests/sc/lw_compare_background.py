"""LTE formal solution (J) with background opacities: dexrt_sc (tables from
dexrt.background_tables) vs Lightweaver (BasicBackground) on a plane-parallel
FAL-C, H_6 (CRTAF) in LTE, zero radiation boundaries.
Usage: python lw_compare_background.py DEXRT_SC TABLES WORKDIR"""
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
import lightweaver as lw
from lightweaver.fal import Falc82

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dexrt_py"))
from lw_models import crtaf_h6
from dexrt.background_tables import _disable_debye, _atom_factories, DEFAULT_ELEMENTS

dexrt_sc = Path(sys.argv[1]).resolve()
tables = Path(sys.argv[2]).resolve()
work = Path(sys.argv[3]).resolve()
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
_disable_debye()

falc = Falc82()
z = np.asarray(falc.height)[::-1]
nz, nx, dz = 128, 32, 20e3
zg = -100e3 + dz * np.arange(nz)
interp = lambda q: np.interp(zg, z, np.asarray(q)[::-1])
T = interp(falc.temperature)
ne = np.exp(interp(np.log(falc.ne)))
nh = np.exp(interp(np.log(falc.nHTot)))
vturb = interp(falc.vturb)

KB = 1.380649e-23
atmos_path = work / "falc_pp.nc"
with ncdf.Dataset(atmos_path, "w") as f:
    f.createDimension("x", nx)
    f.createDimension("z", nz)
    for k, v in dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=(1.1 * nh + ne) * KB * T,
                     vx=0 * T, vy=0 * T, vz=0 * T).items():
        f.createVariable(k, "f4", ("z", "x"))[...] = np.repeat(v[:, None], nx, axis=1)
    f.createVariable("voxel_scale", "f4")[...] = dz


def run_dexrt(name, background):
    conf = {
        "mode": "Lte",
        "atmos_path": str(atmos_path),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Zero",
        "system": {"mem_pool_gb": 4.0},
        "sc": {"quadrature": "Stepan", "quadrature_order": 7, "x_boundary": "Periodic"},
    }
    if background:
        conf["background_path"] = str(tables)
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    with open(work / f"{name}.log", "w") as log:
        res = subprocess.run([str(dexrt_sc), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print((work / f"{name}.log").read_text()[-3000:])
        raise RuntimeError("dexrt_sc failed")
    out = ncdf.Dataset(work / f"{name}.nc")
    return np.asarray(out["wavelength"][...]), np.asarray(out["J"][...])[:, :nz, :nx].mean(axis=2)


def run_lw(background):
    atmos = lw.Atmosphere.make_1d(
        scale=lw.ScaleType.Geometric, depth_scale=zg[::-1].copy(), temperature=T[::-1].copy(),
        vlos=np.zeros(nz), vturb=vturb[::-1].copy(), ne=ne[::-1].copy(), nh_tot=nh[::-1].copy(),
        lower_bc=lw.ZeroRadiation(), upper_bc=lw.ZeroRadiation(),
    )
    atmos.quadrature(5)
    atoms = _atom_factories()
    models = [crtaf_h6()] + ([atoms[e]() for e in DEFAULT_ELEMENTS] if background else [])
    rs = lw.RadiativeSet(models)
    rs.set_detailed_static("H")
    spect = rs.compute_wavelength_grid()
    eq = rs.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq, Nthreads=16)
    if not background:
        ctx.background.eta[...] = 0.0
        ctx.background.chi[...] = 0.0
        ctx.background.sca[...] = 0.0
    ctx.formal_sol_gamma_matrices()
    wl = np.asarray(spect.wavelength)
    J = np.asarray(ctx.spect.J)[:, ::-1]
    J = np.stack([lw.convert_specific_intensity(wl[la], J[la], outUnits="kW / (m2 nm sr)").value for la in range(wl.shape[0])])
    return wl, J


for background in [False, True]:
    wl_d, J_d = run_dexrt("bg" if background else "nobg", background)
    wl_l, J_l = run_lw(background)
    idx_d, idx_l = [], []
    for i, w in enumerate(wl_d):
        j = np.argmin(np.abs(wl_l - w))
        if abs(wl_l[j] - w) < 1e-6 * w:
            idx_d.append(i)
            idx_l.append(j)
    A = J_d[idx_d]
    B = J_l[idx_l]
    # NOTE: Away from the bottom boundary (both are zero radiation).
    rows = slice(20, nz - 2)
    m = B[:, rows] > 1e-10 * B.max()
    r = np.abs(A[:, rows] - B[:, rows])[m] / B[:, rows][m]
    print(f"background {background}: {len(idx_d)} common wavelengths, J rel diff median {np.median(r):.2e}, 90% {np.percentile(r, 90):.2e}, 99% {np.percentile(r, 99):.2e}")
    np.savez(work / f"J_{'bg' if background else 'nobg'}.npz", wl=wl_d[idx_d], J_dex=A, J_lw=B, z=zg)
