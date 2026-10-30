"""Longer characteristics vs the quadtree_ray_traversal prototype (`sweep`,
integer lattice, uniform forest): emergent intensities of the upward
directions on a Gaussian blob (GivenFs, zero boundaries). Both are exact long
characteristics for piecewise-constant material on the same lattice, so they
must agree to rounding. Directions the prototype sweeps along x (|mu_x| >
|mu_z|) use a different lattice and are skipped.
Usage: python sc_longer_vs_prototype.py DEXRT_SC WORKDIR"""
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sc_givenfs_tests_lib import run_givenfs, emergent_dir_indices  # noqa: E402
from quadtree_ray_traversal.amr_traversal import QuadtreeForest  # noqa: E402
from quadtree_ray_traversal.raytrace import CellMaterial  # noqa: E402
from quadtree_ray_traversal.sweep import sweep  # noqa: E402

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)

m, roots = 8, (8, 4)
nx, nz = roots[0] * m, roots[1] * m
dx = 1.0 / m
xc = (np.arange(nx) + 0.5) * dx
zc = (np.arange(nz) + 0.5) * dx
Z, X = np.meshgrid(zc, xc, indexing="ij")
blob = np.exp(-((X - 3.0) ** 2 + (Z - 2.0) ** 2) / (2 * 0.6**2))
chi = (0.3 + 4.0 * blob)[..., None].astype(np.float32)
eta = (chi[..., 0] * (0.5 + 2.0 * blob))[..., None].astype(np.float32)

failures = []
for periodic in [False, True]:
    sc = {"characteristics": "LongerCharacteristics", "quadrature": "Stepan", "quadrature_order": 7,
          "x_boundary": "Periodic" if periodic else "Fixed"}
    out = run_givenfs(dexrt_sc, work, f"longer_blob_{periodic}", eta, chi, sc, voxel_scale=dx)
    I = np.asarray(out["I_emergent"][0])
    mu = np.asarray(out["sc_mu"][...])[emergent_dir_indices(out)]
    phase = np.asarray(out.getncattr("sc_longer_phase_top"))[emergent_dir_indices(out)]
    forest = QuadtreeForest(roots, block_cells=m)
    worst = 0.0
    n_dirs = 0
    for d in range(mu.shape[0]):
        mux, muz = float(mu[d, 0]), float(mu[d, 2])
        if muz <= 0.0 or abs(mux) > abs(muz):
            continue
        n_dirs += 1
        # NOTE: The prototype is 2D: scale the material so the optical depth
        # per unit height matches the 3D ray (path per z: 1/|mu_z|).
        s = 1.0 / math.hypot(mux, muz)
        material = lambda b, c: CellMaterial(float(chi[b.iy * m + c[1], b.ix * m + c[0], 0]) * s,
                                             float(eta[b.iy * m + c[1], b.ix * m + c[0], 0]) * s)
        res = sweep(forest, (mux, muz), material, boundary_intensity=0.0, periodic_x=periodic,
                    periodic_y=False, lattice="integer")
        emer = res.emergent.on("+y")
        pos = np.asarray(emer.position)
        Ip = np.asarray(emer.intensity)
        # NOTE: dexrt labels each by its canonical entry cell s above the top
        # face, at canonical a = s + phase; physical x = x_s + phase, or
        # x_s + 1 - phase when flipped in x.
        offset = phase[d] if mux >= 0 else 1.0 - phase[d]
        x_dex = ((np.arange(nx) + offset) * dx) % (nx * dx)
        pos_w = pos % (nx * dx)
        match = np.array([np.argmin(np.minimum(np.abs(x_dex - p), nx * dx - np.abs(x_dex - p))) for p in pos_w])
        dist = np.abs(x_dex[match] - pos_w)
        dist = np.minimum(dist, nx * dx - dist)
        if dist.max() > 1e-4 * dx:
            failures.append(f"periodic={periodic} dir {d}: positions differ by {dist.max() / dx:.2e} cells")
        I_d = I[d][match]
        valid = Ip > 1e-6 * Ip.max()
        rel = np.abs(I_d[valid] - Ip[valid]) / Ip[valid]
        worst = max(worst, rel.max())
        if rel.max() > 1e-4:
            failures.append(f"periodic={periodic} dir {d} (mu_x {mux:+.3f}, mu_z {muz:.3f}): max rel {rel.max():.2e}")
    print(f"periodic={periodic}: {n_dirs} directions, worst emergent rel diff {worst:.2e}")

for f in failures:
    print("FAIL", f)
if failures:
    sys.exit(1)
print("Longer characteristics match the prototype")
