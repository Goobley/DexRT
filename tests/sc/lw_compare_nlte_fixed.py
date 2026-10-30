"""Non-LTE H_6 with fixed (zero) boundaries on all sides: Lightweaver vs
dexrt_sc vs dexrt (RC) outputs already computed in WORKDIR (sc_fixed_out.nc,
rc_fixed_out.nc). Usage: python lw_compare_nlte_fixed.py WORKDIR"""
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom
sys.path.insert(0, str(Path(__file__).parent))
from lw_models import crtaf_h6
import lightweaver.atomic_set as lw_atomic_set

# NOTE: DexRT does not include Debye shielding in the LTE populations; disable
# it in Lightweaver so the comparison isolates the solvers.
_lte_pops = lw_atomic_set.lte_pops
def _lte_pops_no_debye(*args, **kwargs):
    kwargs["debye"] = False
    return _lte_pops(*args, **kwargs)
lw_atomic_set.lte_pops = _lte_pops_no_debye
for _mod in list(sys.modules.values()):
    if getattr(_mod, "lte_pops", None) is _lte_pops:
        _mod.lte_pops = _lte_pops_no_debye

work = Path(sys.argv[1]).resolve()
nx, nz, dx = 64, 64, 100e3
zz, xx = np.meshgrid((np.arange(nz) + 0.5) / nz, (np.arange(nx) + 0.5) / nx, indexing="ij")
T = 6000.0 + 4000.0 * np.exp(-((xx - 0.4) ** 2 + (zz - 0.55) ** 2) / 0.02) + 1500.0 * zz
nh = 3e16 * (1.0 + 2.0 * np.exp(-((xx - 0.6) ** 2 + (zz - 0.3) ** 2) / 0.03))
ne = 0.2 * nh
vturb = np.full_like(T, 5e3)


class ZeroXBc(lw.BoundaryCondition):
    def compute_bc(self, atmos, spect):
        return np.zeros((spect.wavelength.shape[0], self.mux.shape[0], atmos.Nz))


# NOTE: Lightweaver pins I on its outermost rows/columns: pad all sides with
# near-transparent cells of the same temperature.
def pad(a, scale=1.0):
    a = np.concatenate([a[:1] * scale, a, a[-1:] * scale], axis=0)
    return np.concatenate([a[:, :1] * scale, a, a[:, -1:] * scale], axis=1)


flip = lambda a: np.ascontiguousarray(a[::-1].astype(np.float64))
atmos = lw.Atmosphere.make_2d(
    height=((np.arange(nz + 2) - 0.5) * dx)[::-1].copy(), x=(np.arange(nx + 2) - 0.5) * dx,
    temperature=flip(pad(T)), vx=flip(pad(0 * T)), vz=flip(pad(0 * T)), vturb=flip(pad(vturb)),
    ne=flip(pad(ne, 1e-12)), nh_tot=flip(pad(nh, 1e-12)),
    x_lower_bc=ZeroXBc(), x_upper_bc=ZeroXBc(),
    z_lower_bc=lw.ZeroRadiation(), z_upper_bc=lw.ZeroRadiation(),
)
atmos.quadrature(3)
h_model = crtaf_h6() if "--crtaf" in sys.argv else H_6_atom()
print("Lightweaver H model:", "CRTAF tests/H_6.yaml" if "--crtaf" in sys.argv else "rh_atoms.H_6_atom")
a_set = lw.RadiativeSet([h_model])
a_set.set_active("H")
spect = a_set.compute_wavelength_grid()
eq_pops = a_set.compute_eq_pops(atmos)
ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16, formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
ctx.background.eta[...] = 0.0
ctx.background.chi[...] = 0.0
ctx.background.sca[...] = 0.0
it = lw.iterate_ctx_se(ctx, pops_tol=1e-4, max_iter=1000, quiet=True)
print("LW iterations", it)
lwp = np.asarray(eq_pops["H"]).reshape(-1, nz + 2, nx + 2)[:, ::-1][:, 1:-1, 1:-1]
np.save(work / "pops_lw_fixed.npy", lwp)
sc = np.asarray(ncdf.Dataset(work / "sc_fixed_out.nc")["pops"][...])[:, :nz, :nx]
rc = np.asarray(ncdf.Dataset(work / "rc_fixed_out.nc")["pops"][...])[:, :nz, :nx]
for l in range(lwp.shape[0]):
    rs = np.abs(sc[l] - lwp[l]) / lwp[l]
    rr = np.abs(rc[l] - lwp[l]) / lwp[l]
    print(f"level {l}: SC vs LW median {np.median(rs):.2e} max {rs.max():.2e} | RC vs LW median {np.median(rr):.2e} max {rr.max():.2e}")
