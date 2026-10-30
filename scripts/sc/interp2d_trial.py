"""Trial of the 3D SC plane interpolation (plans/sc_3d.md): in vacuum a 3D
sweep is a repeated plane remap I_p = remap(I_{p-1}, t_x, t_y), so the
interpolator's spread, azimuthal symmetry, order and bounds can be measured
on a periodic plane. Variants: successive 1D remaps (x then y, y then x, the
average of both) and the unsplit reconstruction (limited sx, sy, twist sxy,
Barth-Jespersen corner bound).
Usage: python interp2d_trial.py OUTDIR"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def slope(dm, dp, interp):
    if interp == "Linear":
        return np.zeros_like(dm)
    same = dm * dp > 0.0
    if interp == "VanLeer":
        s = 2.0 * dm * dp / np.where(same, dm + dp, 1.0)
    else:
        mag = np.minimum(np.minimum(np.abs(2.0 * dm), np.abs(2.0 * dp)), 0.5 * np.abs(dm + dp))
        s = np.copysign(mag, dm)
    return np.where(same, s, 0.0)


def sc_interp(fm2, fm1, f0, fp1, t, interp):
    """As ScInterp.hpp: the value a distance t before f0."""
    linear = f0 - t * (f0 - fm1)
    if interp == "Linear":
        return linear
    s0 = slope(f0 - fm1, fp1 - f0, interp)
    sm1 = slope(fm1 - fm2, f0 - fm1, interp)
    return linear - 0.5 * t * (1.0 - t) * (s0 - sm1)


def remap_1d(F, shift, axis, interp):
    n = int(np.floor(shift))
    t = shift - n
    r = lambda k: np.roll(F, n + k, axis=axis)  # r(k)[i] = F[i - n - k]
    return sc_interp(r(2), r(1), r(0), r(-1), t, interp)


def remap_split(F, tx, ty, interp, order):
    if order == "xy":
        return remap_1d(remap_1d(F, tx, 1, interp), ty, 0, interp)
    if order == "yx":
        return remap_1d(remap_1d(F, ty, 0, interp), tx, 1, interp)
    return 0.5 * (remap_split(F, tx, ty, interp, "xy") + remap_split(F, tx, ty, interp, "yx"))


def minmod(a, b):
    return np.where(a * b > 0.0, np.copysign(np.minimum(np.abs(a), np.abs(b)), a), 0.0)


def unsplit_slopes(F, interp):
    """Per-cell (sx, sy, sxy), BJ-bounded against the 3x3 neighbourhood
    (planar part first, then the twist within the remaining headroom)."""
    R = lambda dj, di: np.roll(np.roll(F, dj, axis=0), di, axis=1)  # R[j, i] = F[j - dj, i - di]
    sx = slope(F - R(0, 1), R(0, -1) - F, interp)
    sy = slope(F - R(1, 0), R(-1, 0) - F, interp)
    cx = 0.5 * (R(0, -1) - R(0, 1))
    cy = 0.5 * (R(-1, 0) - R(1, 0))
    sxy = minmod(slope(cx - np.roll(cx, 1, 0), np.roll(cx, -1, 0) - cx, interp),
                 slope(cy - np.roll(cy, 1, 1), np.roll(cy, -1, 1) - cy, interp))
    nb = np.stack([R(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1)])
    lo = nb.min(axis=0) - F
    hi = nb.max(axis=0) - F
    phi = np.ones_like(F)
    corners = [(u, v) for u in (-0.5, 0.5) for v in (-0.5, 0.5)]
    with np.errstate(divide="ignore", invalid="ignore"):
        for u, v in corners:
            d = u * sx + v * sy
            phi = np.where(d > hi, np.minimum(phi, hi / d), phi)
            phi = np.where(d < lo, np.minimum(phi, lo / d), phi)
    sx = sx * phi
    sy = sy * phi
    w = 0.25 * sxy
    w_lo = np.full_like(F, -np.inf)
    w_hi = np.full_like(F, np.inf)
    for u, v in corners:
        d = u * sx + v * sy
        if u * v > 0:
            w_lo, w_hi = np.maximum(w_lo, lo - d), np.minimum(w_hi, hi - d)
        else:
            w_lo, w_hi = np.maximum(w_lo, d - hi), np.minimum(w_hi, d - lo)
    w = np.clip(w, w_lo, w_hi)
    return sx, sy, 4.0 * w


def remap_unsplit(F, tx, ty, interp):
    if interp == "Linear":
        return remap_split(F, tx, ty, "Linear", "xy")
    sx, sy, sxy = unsplit_slopes(F, interp)
    nx, ny = int(np.floor(tx)), int(np.floor(ty))
    rx, ry = tx - nx, ty - ny
    out = np.zeros_like(F)
    # NOTE: cell offset 1 is "cell -1" (weight r, centroid (1 - r)/2), 0 is "cell 0".
    for kx, Wx, xi in ((1, rx, 0.5 * (1.0 - rx)), (0, 1.0 - rx, -0.5 * rx)):
        for ky, Wy, eta in ((1, ry, 0.5 * (1.0 - ry)), (0, 1.0 - ry, -0.5 * ry)):
            sh = lambda A: np.roll(np.roll(A, ny + ky, axis=0), nx + kx, axis=1)
            out += Wx * Wy * (sh(F) + sh(sx) * xi + sh(sy) * eta + sh(sxy) * xi * eta)
    return out


VARIANTS = [
    ("Linear", "xy"),
    ("VanLeer", "xy"), ("VanLeer", "yx"), ("VanLeer", "sym"), ("VanLeer", "unsplit"),
    ("MonotonizedCentral", "xy"), ("MonotonizedCentral", "yx"), ("MonotonizedCentral", "sym"),
    ("MonotonizedCentral", "unsplit"),
]


def step(F, tx, ty, interp, order):
    if order == "unsplit":
        return remap_unsplit(F, tx, ty, interp)
    return remap_split(F, tx, ty, interp, order)


def offsets(N, cx, cy):
    x = np.arange(N)
    return (x[None, :] - cx + N / 2) % N - N / 2, (x[:, None] - cy + N / 2) % N - N / 2


def gaussian(N, cx, cy, sigma):
    dx, dy = offsets(N, cx, cy)
    return np.exp(-0.5 * (dx**2 + dy**2) / sigma**2)


def disc(N, cx, cy, radius):
    """Smooth-edged disc (super-Gaussian): non-separable, symmetric under x <-> y."""
    dx, dy = offsets(N, cx, cy)
    return np.exp(-((dx**2 + dy**2) / radius**2) ** 2)


def ellipse(N, cx, cy, sigma):
    """Gaussian elongated along the diagonal (2:1): smooth, non-separable,
    symmetric under x <-> y."""
    dx, dy = offsets(N, cx, cy)
    u, v = (dx + dy) / np.sqrt(2.0), (dx - dy) / np.sqrt(2.0)
    return np.exp(-0.5 * (u**2 / (2.0 * sigma) ** 2 + v**2 / sigma**2))


def moments(F, cx, cy):
    N = F.shape[0]
    x = np.arange(N)
    dx = ((x[None, :] - cx + N / 2) % N - N / 2) * np.ones_like(F)
    dy = ((x[:, None] - cy + N / 2) % N - N / 2) * np.ones_like(F)
    m = F.sum()
    mx, my = (F * dx).sum() / m, (F * dy).sum() / m
    vxx = (F * (dx - mx) ** 2).sum() / m
    vyy = (F * (dy - my) ** 2).sum() / m
    vxy = (F * (dx - mx) * (dy - my)).sum() / m
    return mx, my, vxx, vyy, vxy


def searchlight(outdir):
    """Narrow beam, theta = 60 deg, 48 planes: added variance along x and y
    against azimuth. A scheme symmetric under x <-> y gives
    dVxx(phi) = dVyy(90 - phi) and dVxy(phi) = dVxy(90 - phi)."""
    N, radius, planes = 160, 3.0, 48
    tan_theta = np.tan(np.radians(60.0))
    phis = np.arange(0.0, 90.01, 7.5)
    results = {}
    for interp, order in VARIANTS:
        rows = []
        for phi in phis:
            tx, ty = tan_theta * np.cos(np.radians(phi)), tan_theta * np.sin(np.radians(phi))
            F0 = disc(N, N / 4, N / 4, radius)
            _, _, v0, _, _ = moments(F0, N / 4, N / 4)
            F = F0
            for _ in range(planes):
                F = step(F, tx, ty, interp, order)
            ex, ey = N / 4 + planes * tx, N / 4 + planes * ty
            exact = disc(N, ex, ey, radius)
            mx, my, vxx, vyy, vxy = moments(F, ex, ey)
            rows.append((mx, my, vxx - v0, vyy - v0, vxy, F.min() / F.max(), np.abs(F - exact).sum() / exact.sum()))
        results[(interp, order)] = np.array(rows)

    print("\nSearchlight (disc r = 3, theta = 60 deg, 48 planes): added variance [cells^2]")
    print(f"{'variant':28s} {'mean dV':>8s} {'max |centroid|':>14s} {'asym Vxx/Vyy':>13s} {'asym Vxy':>9s} {'min/max':>9s} {'mean L1':>8s}")
    for key, r in results.items():
        asym = np.max(np.abs(r[:, 2] - r[::-1, 3]))
        asym_xy = np.max(np.abs(r[:, 4] - r[::-1, 4]))
        cen = np.max(np.hypot(r[:, 0], r[:, 1]))
        print(f"{key[0] + ' ' + key[1]:28s} {np.mean(r[:, 2] + r[:, 3]):8.3f} {cen:14.2e} {asym:13.2e} {asym_xy:9.2e} {r[:, 5].min():9.1e} {r[:, 6].mean():8.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key, r in results.items():
        label = f"{key[0][:2]} {key[1]}"
        ls = "-" if key[1] in ("xy", "unsplit") else "--" if key[1] == "yx" else ":"
        axes[0].plot(phis, r[:, 2], ls, label=label)
        axes[1].plot(phis, r[:, 3], ls, label=label)
        axes[2].plot(phis, r[:, 4], ls, label=label)
    for ax, title in zip(axes, ["added Vxx", "added Vyy", "Vxy"]):
        ax.set_xlabel("azimuth phi [deg]")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "searchlight_spread.png", dpi=120)
    return results


def convergence(outdir):
    """Smooth diagonal ellipse (sigma = N/24) carried a fixed distance (N/2 cells in
    t per plane), refining N: L1 error against the exact translate."""
    print("\nConvergence (smooth diagonal ellipse, fixed path in domain units): L1 error, order")
    Ns = [32, 64, 128, 256]
    tx, ty = 0.37, 0.71
    errs = {}
    for interp, order in VARIANTS:
        e = []
        for N in Ns:
            planes = N  # path ~ N * |t| cells, i.e. fixed in domain units
            sigma = N / 24
            F = ellipse(N, N / 2, N / 2, sigma)
            for _ in range(planes):
                F = step(F, tx, ty, interp, order)
            exact = ellipse(N, N / 2 + planes * tx, N / 2 + planes * ty, sigma)
            e.append(np.abs(F - exact).sum() / np.abs(exact).sum())
        errs[(interp, order)] = e
        orders = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
        print(f"{interp + ' ' + order:28s} " + " ".join(f"{v:.2e}" for v in e) + "   order " + " ".join(f"{o:.2f}" for o in orders))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, e in errs.items():
        ax.loglog(Ns, e, "o-", label=f"{key[0][:2]} {key[1]}")
    ax.set_xlabel("N")
    ax.set_ylabel("relative L1 error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "convergence.png", dpi=120)


def bounds():
    """Top-hat square and a diagonal step: min and max after 40 planes."""
    print("\nBounds (initial data in [0, 1]): min, max after 40 planes at (tx, ty) = (0.6, 0.45)")
    N = 64
    x = np.arange(N)
    hat = ((np.abs(x[None, :] - 20) < 6) & (np.abs(x[:, None] - 20) < 6)).astype(float)
    diag = (x[None, :] + x[:, None] < N).astype(float)
    checker = ((x[None, :] // 2 + x[:, None] // 2) % 2).astype(float)
    for name, F0 in (("top-hat", hat), ("diagonal step", diag), ("checkerboard", checker)):
        line = f"{name:14s}"
        for interp, order in VARIANTS:
            F = F0.copy()
            for _ in range(40):
                F = step(F, 0.6, 0.45, interp, order)
            line += f"  {interp[:2]} {order}: [{F.min():.1e}, {F.max():.4f}]"
        print(line)


def limits():
    """ty = 0 reduces to the 1D remap; Linear is bilinear; bilinear data exact."""
    rng = np.random.default_rng(1)
    F = rng.random((32, 32))
    print("\nLimits")
    for interp in ("Linear", "VanLeer", "MonotonizedCentral"):
        one_d = remap_1d(F, 2.3, 1, interp)
        print(f"  {interp:20s} ty=0 vs 1D: split {np.max(np.abs(remap_split(F, 2.3, 0.0, interp, 'xy') - one_d)):.1e}"
              f"  unsplit {np.max(np.abs(remap_unsplit(F, 2.3, 0.0, interp) - one_d)):.1e}")
    x = np.arange(32.0)
    lin = 0.1 * x[None, :] + 0.2 * x[:, None] + 3.0
    exact = 0.1 * (x[None, :] - 0.3) + 0.2 * (x[:, None] - 0.6) + 3.0
    interior = (slice(4, -4), slice(4, -4))
    for interp in ("VanLeer", "MonotonizedCentral"):
        for order in ("xy", "unsplit"):
            out = step(lin, 0.3, 0.6, interp, order)
            print(f"  {interp:20s} {order:8s} linear data error (interior): {np.max(np.abs(out - exact)[interior]):.1e}")


def order_dependence():
    """Direct x-then-y vs y-then-x difference on non-separable data."""
    print("\nOrder dependence: max |xy - yx| / max after 48 planes, (tx, ty) = (1.3, 0.75)")
    N = 96
    for name, F0 in (("disc r=3", disc(N, 30, 30, 3.0)), ("ellipse", ellipse(N, 30, 30, 3.0))):
        line = f"  {name:10s}"
        for interp in ("VanLeer", "MonotonizedCentral"):
            A, B = F0.copy(), F0.copy()
            for _ in range(48):
                A = step(A, 1.3, 0.75, interp, "xy")
                B = step(B, 1.3, 0.75, interp, "yx")
            line += f"  {interp}: {np.max(np.abs(A - B)) / A.max():.2e}"
        print(line)


if __name__ == "__main__":
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    limits()
    bounds()
    order_dependence()
    searchlight(outdir)
    convergence(outdir)
