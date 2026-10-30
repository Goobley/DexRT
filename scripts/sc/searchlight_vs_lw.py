"""Searchlight: dexrt_sc vs Lightweaver on identical given eta/chi (Lightweaver
background, negligible-density H_6), mirrored Stepan quadrature (the same ray
set in both), zero boundaries. Plots the emergent beam at the top for each
upward direction against the exact (shifted top hat) result.
Usage: python searchlight_vs_lw.py DEXRT_SC WORKDIR"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "sc"))
from sc_givenfs_tests_lib import run_givenfs, emergent_dir_indices

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)

nx, nz = 512, 64
dx = 1e5
x0, w, z_src = 252, 8, 1
chi_bg = 1e-20 / dx
chi = np.full((nz, nx), chi_bg)
chi[z_src, x0:x0 + w] = 1.0 / dx
S = np.zeros_like(chi)
S[z_src, x0:x0 + w] = 1.0
eta = chi * S


class ZeroXBc(lw.BoundaryCondition):
    def compute_bc(self, atmos, spect):
        return np.zeros((spect.wavelength.shape[0], self.mux.shape[0], atmos.Nz))


def run_lw(interp):
    flip = lambda a: np.ascontiguousarray(a[::-1].astype(np.float64))
    atmos = lw.Atmosphere.make_2d(
        height=((np.arange(nz) + 0.5) * dx)[::-1].copy(), x=(np.arange(nx) + 0.5) * dx,
        temperature=np.full((nz, nx), 5000.0), vx=np.zeros((nz, nx)), vz=np.zeros((nz, nx)),
        vturb=np.full((nz, nx), 2e3), ne=np.full((nz, nx), 1.0), nh_tot=np.full((nz, nx), 1.0),
        x_lower_bc=ZeroXBc(), x_upper_bc=ZeroXBc(),
        z_lower_bc=lw.ZeroRadiation(), z_upper_bc=lw.ZeroRadiation(),
    )
    atmos.quadrature(3)
    a_set = lw.RadiativeSet([H_6_atom()])
    a_set.set_detailed_static("H")
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16, formal_solver="piecewise_linear_2d", interp_fn=interp)
    ctx.background.chi[...] = flip(chi).reshape(-1)[None, :]
    ctx.background.eta[...] = flip(eta).reshape(-1)[None, :]
    ctx.background.sca[...] = 0.0
    ctx.formal_sol_gamma_matrices(lambda_iterate=True)
    la = np.argmin(np.abs(np.asarray(spect.wavelength) - 500.0))
    mu = np.stack([np.asarray(atmos.mux), np.asarray(atmos.muz)], axis=1)
    return mu, np.asarray(ctx.spect.I)[la]  # [ray (toObs, upward), x] at the top row


lw_results = {"Lightweaver linear": run_lw("interp_linear_2d"), "Lightweaver Bézier": run_lw("interp_besser_2d")}
sc_results = {}
for interp, label in [("Linear", "dexrt_sc linear"), ("MonotonizedCentral", "dexrt_sc MC")]:
    sc = {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear", "interpolation": interp}
    out = run_givenfs(dexrt_sc, work, f"slvlw_{interp}", (eta * dx)[..., None].astype(np.float32),
                      (chi * dx)[..., None].astype(np.float32), sc, voxel_scale=1.0)
    sc_results[label] = (np.asarray(out["sc_mu"][...])[emergent_dir_indices(out)][:, [0, 2]], np.asarray(out["I_emergent"][0]))

xc = np.arange(nx) + 0.5
# NOTE: The +-mu_x directions are mirror images: show mu_x > 0, by obliquity.
mu_lw = lw_results["Lightweaver linear"][0]
dirs = sorted([d for d in range(mu_lw.shape[0]) if mu_lw[d, 0] > 0], key=lambda d: mu_lw[d, 0] / mu_lw[d, 1])
colors = {"dexrt_sc linear": "#2a78d6", "dexrt_sc MC": "#eb6834", "Lightweaver linear": "#1baf7a", "Lightweaver Bézier": "#eda100"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
stats = []
for ax, d in zip(axes.flat, dirs):
    mx, mz = lw_results["Lightweaver linear"][0][d]
    shift = (nz - 1 - z_src) * mx / mz
    centre = x0 + 0.5 * w + shift
    exact = np.zeros(nx)
    lo, hi = x0 + shift, x0 + w + shift
    exact = np.clip(np.minimum(xc + 0.5, hi) - np.maximum(xc - 0.5, lo), 0, 1)
    # NOTE: Shapes are compared at unit area (the nodal emitter's flux isn't a
    # continuous slab's); total flux is reported relative to dexrt_sc linear.
    ax.step(xc, exact / exact.sum(), where="mid", color="#9a9a94", lw=2, label="exact (shifted top hat)")
    flux_ref = None
    row = [f"mu_x={mx:+.2f}, mu_z={mz:.2f}"]
    for label, (mu, I) in {**sc_results, **lw_results}.items():
        k = np.argmin(np.abs(mu[:, 0] - mx) + np.abs(mu[:, 1] - mz))
        prof = I[k]
        tot = prof.sum()
        flux_ref = tot if flux_ref is None else flux_ref
        ax.plot(xc, prof / tot, color=colors[label], lw=2, label=label)
        c = (prof * xc).sum() / tot
        s = np.sqrt((prof * (xc - c) ** 2).sum() / tot)
        row.append(f"{label}: centroid {c - centre:+.2f}, width {s:.2f}, flux {tot / flux_ref:.3f}")
    stats.append(row)
    ax.set_xlim(centre - 30, centre + 30)
    ax.set_title(f"μx = {mx:+.2f}, μz = {mz:.2f}  (shift {shift:+.0f} cells)", fontsize=10)
    ax.set_xlabel("x [cells]")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Emergent I, normalised to unit area")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle(f"Searchlight: {w}-cell emitter at z = {z_src}, {nz - 1 - z_src} rows of vacuum (exact σ = {w / np.sqrt(12):.2f} cells)", fontsize=11)
fig.savefig(work / "searchlight_vs_lw.png", dpi=130)
for row in stats:
    print(row[0])
    for r in row[1:]:
        print("   ", r)
print("saved", work / "searchlight_vs_lw.png")
