"""Searchlight: narrow emitter at the bottom of a vacuum domain. For each
upward direction, the emergent beam centroid at the top must follow the ray
exactly; its width measures the numerical diffusion of each interpolation.
Usage: python sc_searchlight.py DEXRT_SC WORKDIR"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from sc_givenfs_tests_lib import run_givenfs

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)

nx, nz = 256, 128
x0, w = 120, 8
z_src = 1
failures = []
# Non-uniform z grids (dz = None): geometric stretching, and a 3x jump.
# Both span the uniform grid's height.
z_geom = np.cumsum(np.full(nz, 1.02) ** np.arange(nz))
z_geom *= nz / z_geom[-1]
z_jump = np.cumsum(np.where(np.arange(nz) < nz // 2, 1.0, 3.0))
z_jump *= nz / z_jump[-1]
grids = [(1.0, 1.0, None), (2.0, 1.0, None), (1.0, 3.0, None), (1.0, None, z_geom), (1.0, None, z_jump)]
for (dx, dz, z) in grids:
    chi = np.zeros((nz, nx, 1), dtype=np.float32)
    chi[z_src, x0:x0 + w, 0] = 1.0
    eta = chi.copy()
    label = f"{dz}" if z is None else f"non-uniform {z[1] - z[0]:.2f} .. {z[-1] - z[-2]:.2f}"
    tag = f"{dz}" if z is None else f"z_{z[-1] - z[-2]:.1f}"
    print(f"-- dx = {dx}, dz = {label}")
    for interp in ["Linear", "VanLeer", "MonotonizedCentral", "LongerCharacteristics"]:
        if interp == "LongerCharacteristics":
            sc = {"characteristics": "LongerCharacteristics", "quadrature": "Stepan", "quadrature_order": 7}
        else:
            sc = {"characteristics": "Short", "formal_solver": "PiecewiseLinear", "interpolation": interp, "quadrature": "Stepan", "quadrature_order": 7}
        out = run_givenfs(dexrt_sc, work, f"searchlight_{dx}_{tag}_{interp}", eta, chi, sc, voxel_scale=1.0, dx=dx, dz=dz, z=z)
        I = np.asarray(out["I_emergent"][0])
        mu = np.asarray(out["sc_mu"][...])
        xc = (np.arange(nx) + 0.5) * dx
        if interp == "LongerCharacteristics":
            # NOTE: Emergent values sit at the characteristics' exit points.
            phase = np.asarray(out.getncattr("sc_longer_phase_top"))
        src_c = (x0 + 0.5 * w) * dx
        widths = []
        for d in range(mu.shape[0]):
            if mu[d, 2] <= 0:
                continue
            prof = I[d]
            total = prof.sum()
            if interp == "LongerCharacteristics":
                xc = (np.arange(nx) + (phase[d] if mu[d, 0] >= 0 else 1.0 - phase[d])) * dx
            height = (nz - 1 - z_src) * dz if z is None else z[-1] - z[z_src]
            if interp == "LongerCharacteristics":
                # NOTE: These leave through the top face, half a step above the top node.
                height += 0.5 * (dz if z is None else z[-1] - z[-2])
            shift = height * mu[d, 0] / mu[d, 2]
            expected = src_c + shift
            if total <= 0 or not (w * dx < expected < (nx - w) * dx):
                continue
            centroid = (prof * xc).sum() / total
            sigma = np.sqrt((prof * (xc - centroid) ** 2).sum() / total)
            ok = abs(centroid - expected) < 0.5 * dx
            if not ok:
                failures.append(f"{dx},{dz},{interp},dir {d}: centroid {centroid:.2f} expected {expected:.2f}")
            widths.append(sigma / dx)
        print(f"  {interp:20s}: beams {len(widths)}, mean width {np.mean(widths):.2f} cells (source {w / np.sqrt(12):.2f}), max {np.max(widths):.2f}")

for f in failures:
    print("FAIL", f)
if failures:
    sys.exit(1)
print("Searchlight centroids all correct")
