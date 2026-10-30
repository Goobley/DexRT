"""dexrt_sc_3d GivenFs tests.
Usage: python sc_3d_givenfs_tests.py DEXRT_SC_3D WORKDIR [--launcher "mpirun -n 2"]
    [--characteristics Short|LongerCharacteristics]
 - thin: optically thin uniform emission, J against the analytic path sum.
 - thick: J -> S in an optically thick medium.
 - shift: periodic x/y shifts of the material shift J.
 - box (Longer): emergent intensity of each lattice line through an emitting
   box in vacuum against its analytic path length through the box (exact for
   constant material), on cubic, anisotropic and stretched-z grids.
 - box flux (Short, each interpolation): with periodic x/y the emergent
   intensity summed over the top plane against eta V / (dx dy |mu_z|), and
   the beam's added variance against the longer characteristics.
 - healpix (in-plane rays, mu_z = 0): a uniform medium against the analytic J
   (in-plane rays see an infinite homogeneous line, I = S; swept ones the
   analytic node value or cell mean), periodic shifts, fixed side faces.
"""
import argparse
import shlex
import subprocess
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml

# NOTE: J is stored in 16 bits (2^-11 relative), re-rounded per direction batch.
RAD_RTOL = 2e-3


def write_atmos(path, eta, chi, voxel_scale=1.0, dx=None, dy=None, dz=None, z=None):
    nz, ny, nx, nw = eta.shape
    with ncdf.Dataset(path, "w") as f:
        f.createDimension("x", nx)
        f.createDimension("y", ny)
        f.createDimension("z", nz)
        f.createDimension("wavelength", nw)
        f.createVariable("eta", "f4", ("z", "y", "x", "wavelength"))[...] = eta
        f.createVariable("chi", "f4", ("z", "y", "x", "wavelength"))[...] = chi
        f.createVariable("voxel_scale", "f4")[...] = voxel_scale
        for name, v in (("voxel_scale_x", dx), ("voxel_scale_y", dy), ("voxel_scale_z", dz)):
            if v is not None:
                f.createVariable(name, "f4")[...] = v
        if z is not None:
            f.createVariable("z", "f8", ("z",))[...] = z


def run(binary, work, name, eta, chi, sc, launcher=(), output=None, **kw):
    atmos = work / f"{name}_atmos.nc"
    out = work / f"{name}_out.nc"
    write_atmos(atmos, eta, chi, **kw)
    conf = {
        "mode": "GivenFs",
        "atmos_path": str(atmos),
        "output_path": str(out),
        "system": {"mem_pool_gb": 2.0},
        "sc": {"characteristics": CHARACTERISTICS, **sc},
    }
    if output:
        conf["output"] = output
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    log = work / f"{name}.log"
    with open(log, "w") as f:
        r = subprocess.run([*launcher, str(binary), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(log.read_text()[-3000:])
        raise RuntimeError(f"{name} failed")
    return ncdf.Dataset(out)


CHARACTERISTICS = "LongerCharacteristics"


def heights(nz, dz, z):
    """Cell centres and the bottom/top faces (mirrored half steps)."""
    if z is None:
        zc = (np.arange(nz) + 0.5) * dz
    else:
        zc = np.asarray(z, dtype=np.float64)
    return zc, zc[0] - 0.5 * (zc[1] - zc[0]), zc[-1] + 0.5 * (zc[-1] - zc[-2])


def test_thin(binary, work, launcher):
    nz, ny, nx = 24, 16, 16
    eta = np.full((nz, ny, nx, 1), 2.0, dtype=np.float32)
    chi = np.full_like(eta, 1e-9)
    out = run(binary, work, "thin", eta, chi, {"x_boundary": "Periodic", "y_boundary": "Periodic"}, launcher)
    J = np.asarray(out["J"][0])
    mu = np.asarray(out["sc_mu"][...])
    w = np.asarray(out["sc_wmu"][...])
    zc, z0, z1 = heights(nz, 1.0, None)
    # NOTE: the cell-mean distance to the inflow face along the ray
    expected = np.zeros(nz)
    for m, wt in zip(mu, w):
        dist = (zc - z0) if m[2] > 0 else (z1 - zc)
        expected += wt * 2.0 * dist / abs(m[2])
    err = np.max(np.abs(J - expected[:, None, None]) / expected[:, None, None])
    print(f"thin: max rel J error {err:.2e}")
    assert err < RAD_RTOL, err


def test_thick(binary, work, launcher):
    nz, ny, nx = 24, 16, 16
    chi = np.full((nz, ny, nx, 1), 50.0, dtype=np.float32)
    eta = 3.0 * chi
    out = run(binary, work, "thick", eta, chi, {"x_boundary": "Periodic", "y_boundary": "Periodic"}, launcher)
    J = np.asarray(out["J"][0])
    err = np.max(np.abs(J[4:-4] - 3.0) / 3.0)
    print(f"thick: max rel |J - S| (interior) {err:.2e}")
    assert err < RAD_RTOL, err


def test_shift(binary, work, launcher):
    nz, ny, nx = 16, 16, 24
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    blob = np.exp(-((xx - 8.0) ** 2 / 8 + (yy - 5.0) ** 2 / 6 + (zz - 8.0) ** 2 / 10))
    chi = (0.05 + 0.5 * blob)[..., None].astype(np.float32)
    eta = (chi * (1.0 + 4.0 * blob[..., None])).astype(np.float32)
    sc = {"x_boundary": "Periodic", "y_boundary": "Periodic"}
    a = np.asarray(run(binary, work, "shift_a", eta, chi, sc, launcher)["J"][0])
    sx, sy = 7, 3
    b = np.asarray(run(binary, work, "shift_b", np.roll(eta, (sy, sx), (1, 2)), np.roll(chi, (sy, sx), (1, 2)), sc, launcher)["J"][0])
    err = np.max(np.abs(np.roll(a, (sy, sx), (1, 2)) - b) / np.max(a))
    print(f"shift: max |J shifted - J| / max {err:.2e}")
    # Short's fp32 interpolation followed by 16-bit J storage is not bitwise
    # translation invariant. The original CPU implementation gives ~1.5e-4
    # here; require agreement below the storage precision rather than 1e-5.
    tolerance = 5e-4 if CHARACTERISTICS == "Short" else 1e-5
    assert err < tolerance, err


def box_path(p_top, d, lo, hi):
    """Path length of the ray ending at p_top travelling along d (unit,
    upwards) inside the box [lo, hi], traced back to the bottom face z = lo_z_dom."""
    t0, t1 = -np.inf, np.inf
    for k in range(3):
        if abs(d[k]) < 1e-300:
            if p_top[k] < lo[k] or p_top[k] > hi[k]:
                return 0.0
            continue
        a = (lo[k] - p_top[k]) / d[k]
        b = (hi[k] - p_top[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    t1 = min(t1, 0.0)
    return max(t1 - t0, 0.0)


def test_box(
    binary,
    work,
    launcher,
    name,
    dx=1.0,
    dy=1.0,
    dz=1.0,
    z=None,
    quad_order=7,
    box_chi=None,
):
    """Compare emerging rays with the analytic intensity of a uniform box."""
    nz, ny, nx = 20, 24, 28
    zc, z_bot, z_top = heights(nz, dz, z)
    eta = np.zeros((nz, ny, nx, 1), dtype=np.float32)
    chi = np.full_like(eta, 1e-12)
    # NOTE: the box spans whole cells: x cells [9, 17), y [8, 15), z [6, 13)
    ix, iy, iz = (9, 17), (8, 15), (6, 13)
    eta[iz[0]:iz[1], iy[0]:iy[1], ix[0]:ix[1]] = 1.0
    if box_chi is not None:
        chi[iz[0]:iz[1], iy[0]:iy[1], ix[0]:ix[1]] = box_chi
    zf = np.concatenate([[z_bot], 0.5 * (zc[1:] + zc[:-1]), [z_top]])
    lo = np.array([ix[0] * dx, iy[0] * dy, zf[iz[0]]])
    hi = np.array([ix[1] * dx, iy[1] * dy, zf[iz[1]]])
    kw = dict(dx=dx, dy=dy, dz=dz, z=z)
    out = run(binary, work, name, eta, chi, {"quadrature_order": quad_order}, launcher, output={"sc_emergent": True}, **kw)
    I = np.asarray(out["I_emergent"][0])
    mu = np.asarray(out["sc_mu"][...])
    px = np.asarray(out.getncattr("sc_longer_phase_top"))
    py = np.asarray(out.getncattr("sc_longer_phase_top_y"))
    worst = 0.0
    for d in range(mu.shape[0]):
        if mu[d, 2] <= 0:
            continue
        m = mu[d] / np.linalg.norm(mu[d])
        # NOTE: line labelled by its canonical entry cell s above the top face,
        # at canonical position s + phase - 1/2 (node units); stored at the
        # physical index of s.
        xi = np.arange(nx)
        yi = np.arange(ny)
        ox = (px[d] - 0.5) if m[0] >= 0 else -(px[d] - 0.5)
        oy = (py[d] - 0.5) if m[1] >= 0 else -(py[d] - 0.5)
        expected = np.zeros((ny, nx))
        for j in yi:
            for i in xi:
                p = np.array([(i + ox + 0.5) * dx, (j + oy + 0.5) * dy, z_top])
                path = box_path(p, m, lo, hi)
                expected[j, i] = (
                    path if box_chi is None else -np.expm1(-box_chi * path) / box_chi
                )
        scale = np.max(expected) if np.max(expected) > 0 else 1.0
        worst = max(worst, np.max(np.abs(I[d] - expected)) / scale)
    print(f"box {name}: max |I - analytic I| / max {worst:.2e}")
    assert worst < 1e-4, worst


def test_box_flux(binary, work, launcher):
    """Short: flux conservation through the periodic domain, and the spread
    of each upward direction's beam (added variance [cells^2] of the emergent
    intensity about its centroid, against the longer characteristics)."""
    global CHARACTERISTICS
    nz, ny, nx = 24, 32, 32
    eta = np.zeros((nz, ny, nx, 1), dtype=np.float32)
    chi = np.full_like(eta, 1e-12)
    eta[2:5, 14:17, 14:17] = 1.0
    V = 27.0
    sc = {"x_boundary": "Periodic", "y_boundary": "Periodic", "formal_solver": "PiecewiseLinear"}

    def moments(I):
        # NOTE: periodic: centre on the maximum before taking moments.
        j, i = np.unravel_index(np.argmax(I), I.shape)
        I = np.roll(I, (ny // 2 - j, nx // 2 - i), (0, 1))
        y, x = np.mgrid[0:ny, 0:nx]
        m = I.sum()
        cx, cy = (I * x).sum() / m, (I * y).sum() / m
        return m, (I * ((x - cx) ** 2 + (y - cy) ** 2)).sum() / m

    saved = CHARACTERISTICS
    CHARACTERISTICS = "LongerCharacteristics"
    ref = run(binary, work, "flux_longer", eta, chi, sc, launcher, output={"sc_emergent": True})
    I_ref = np.asarray(ref["I_emergent"][0])
    mu = np.asarray(ref["sc_mu"][...])
    CHARACTERISTICS = "Short"
    for interp in ("Linear", "VanLeer", "MonotonizedCentral"):
        out = run(binary, work, f"flux_{interp}", eta, chi, {**sc, "interpolation": interp}, launcher, output={"sc_emergent": True})
        I = np.asarray(out["I_emergent"][0])
        flux_err = 0.0
        spread = []
        for d in range(mu.shape[0]):
            if mu[d, 2] <= 0:
                continue
            m, v = moments(I[d])
            m_ref, v_ref = moments(I_ref[d])
            flux_err = max(flux_err, abs(m - V / abs(mu[d, 2])) / (V / abs(mu[d, 2])))
            spread.append(v - v_ref)
        print(f"box flux {interp}: max rel flux error {flux_err:.2e}, added variance vs longer "
              f"mean {np.mean(spread):.3f} min {np.min(spread):.3f} max {np.max(spread):.3f}")
        assert flux_err < 2e-3, flux_err
    CHARACTERISTICS = saved


def test_healpix(binary, work, launcher):
    """In-plane rays: Healpix (an equatorial ring at mu_z = 0)."""
    nz, ny, nx = 16, 32, 24
    hp = {"quadrature": "Healpix", "quadrature_order": 2}
    for chi0 in (0.01, 0.3):
        eta = np.full((nz, ny, nx, 1), 2.0 * chi0, dtype=np.float32)
        chi = np.full_like(eta, chi0)
        out = run(binary, work, f"hp_uniform_{chi0}", eta, chi, {**hp, "x_boundary": "Periodic", "y_boundary": "Periodic"}, launcher)
        J = np.asarray(out["J"][0])
        mu = np.asarray(out["sc_mu"][...])
        w = np.asarray(out["sc_wmu"][...])
        zc = np.arange(nz) + 0.5
        expected = np.zeros(nz)
        for m, wt in zip(mu, w):
            if abs(m[2]) < 1e-6:
                expected += wt * 2.0
                continue
            s_c = (zc if m[2] > 0 else nz - zc) / abs(m[2])
            if CHARACTERISTICS == "LongerCharacteristics":
                # NOTE: the cell mean along the ray.
                a = chi0 / (2 * abs(m[2]))
                expected += wt * 2.0 * (1 - np.exp(-chi0 * s_c) * np.sinh(a) / a)
            else:
                expected += wt * 2.0 * (1 - np.exp(-chi0 * s_c))
        err = np.max(np.abs(J - expected[:, None, None]) / expected[:, None, None])
        print(f"healpix uniform chi={chi0}: {np.sum(np.abs(mu[:, 2]) < 1e-6)} in-plane dirs, max rel J error {err:.2e}")
        assert err < RAD_RTOL, err
    # NOTE: shifts (periodic), and fixed side faces (finite, non-negative).
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    blob = np.exp(-((xx - 8.0) ** 2 / 8 + (yy - 5.0) ** 2 / 6 + (zz - 8.0) ** 2 / 10))
    chi = (0.05 + 0.5 * blob)[..., None].astype(np.float32)
    eta = (chi * (1.0 + 4.0 * blob[..., None])).astype(np.float32)
    sc = {**hp, "x_boundary": "Periodic", "y_boundary": "Periodic"}
    a = np.asarray(run(binary, work, "hp_shift_a", eta, chi, sc, launcher)["J"][0])
    sx, sy = 7, 3
    b = np.asarray(run(binary, work, "hp_shift_b", np.roll(eta, (sy, sx), (1, 2)), np.roll(chi, (sy, sx), (1, 2)), sc, launcher)["J"][0])
    err = np.max(np.abs(np.roll(a, (sy, sx), (1, 2)) - b) / np.max(a))
    print(f"healpix shift: max |J shifted - J| / max {err:.2e}")
    # NOTE: the in-plane lattice is anchored to the first face of its primary
    # axis, so shifts along it resample the cells (discretisation level).
    assert err < 1e-2, err
    for xb, yb in (("Fixed", "Fixed"), ("Fixed", "Periodic"), ("Periodic", "Fixed")):
        J = np.asarray(run(binary, work, f"hp_{xb}_{yb}", eta, chi, {**hp, "x_boundary": xb, "y_boundary": yb}, launcher)["J"][0])
        assert np.all(np.isfinite(J)) and np.min(J) >= 0.0, (xb, yb)
        # NOTE: zero incident radiation: below the periodic J everywhere.
        assert np.all(J <= a * (1 + RAD_RTOL)), (xb, yb, np.max(J / a))
        print(f"healpix x {xb}, y {yb}: J in [{np.min(J):.3e}, {np.max(J):.3e}], max J / J_periodic {np.max(J / a):.4f}")


def main():
    global CHARACTERISTICS
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("work")
    ap.add_argument("--launcher", default="")
    ap.add_argument("--characteristics", default="LongerCharacteristics")
    args = ap.parse_args()
    CHARACTERISTICS = args.characteristics
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    launcher = shlex.split(args.launcher)
    binary = Path(args.binary).resolve()
    test_thin(binary, work, launcher)
    test_thick(binary, work, launcher)
    test_shift(binary, work, launcher)
    if CHARACTERISTICS == "LongerCharacteristics":
        test_box(binary, work, launcher, "box_cubic")
        test_box(binary, work, launcher, "box_amplifying", box_chi=-0.5)
        test_box(binary, work, launcher, "box_aniso", dx=1.3, dy=0.7, dz=0.9)
        z = np.cumsum(np.linspace(0.6, 1.5, 20))
        test_box(binary, work, launcher, "box_stretched", z=z)
        test_box(binary, work, launcher, "box_il15", quad_order=15)
    else:
        test_box_flux(binary, work, launcher)
    test_healpix(binary, work, launcher)
    print("All 3D GivenFs tests passed")


if __name__ == "__main__":
    main()
