import numpy as np
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

QuadType = "BasicTrapezoidal"
QuadType = "OctahedralPbrt"
# QuadType = "ClassicOctahedral"
# QuadType = "Healpix"

# TODO(cmo): Develop own version of nest2vec. See how efficient it can be. https://github.com/ntessore/healpix/blob/60af7fff259f4dcaebbb180f2f794e78ec4a9e41/src/healpix.c#L432
# Also look into TOAST: https://iopscience.iop.org/article/10.3847/1538-4365/aaf79e/pdf

if QuadType == "Healpix":
    import healpix

def basic_dir(phi_idx, theta_idx, n_phi, n_theta):
    phi = 2 * np.pi / n_phi * (phi_idx + 0.5)
    cos_theta = 2.0 / n_theta * (theta_idx + 0.5) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    d = np.array([
        np.cos(phi) * sin_theta,
        np.sin(phi) * sin_theta,
        cos_theta
    ]).T
    return np.ascontiguousarray(d)

def classic_oct_dir(phi_idx, theta_idx, n_phi, n_theta):
    phi01 = (phi_idx + 0.5) / n_phi
    theta01 = (theta_idx + 0.5) / n_theta
    u = 2.0 * phi01 - 1.0
    v = 2.0 * theta01 - 1.0
    d = np.array([
        u, v, 1.0 - np.abs(u) - np.abs(v)
    ])
    d[:2, :] = np.where(d[2, :] < 0.0, (1.0 - np.abs(d[[1, 0], :])) * np.copysign(1.0, d[:2, :]), d[:2, :])
    d = np.ascontiguousarray(d.T)
    d_norm = np.sqrt(np.sum(d**2, axis=1))
    d /= d_norm[:, None]
    return d

def clarberg_pbrt_dir(phi_idx, theta_idx, n_phi, n_theta):
    phi01 = (phi_idx + 0.5) / n_phi
    theta01 = (theta_idx + 0.5) / n_theta
    u = 2.0 * phi01 - 1.0
    v = 2.0 * theta01 - 1.0
    up = np.abs(u)
    vp = np.abs(v)

    signed_distance = 1.0 - (up + vp)
    d = np.abs(signed_distance)
    r = 1.0 - d
    phi = np.where(r == 0, 1, (vp - up) / r + 1) * 0.25 * np.pi
    # phi = ((vp - up) / r + 1) * 0.25 * np.pi
    z = np.copysign(1 - r**2, signed_distance)
    cos_phi = np.copysign(np.cos(phi), u)
    sin_phi = np.copysign(np.sin(phi), v)

    d = np.array([
        cos_phi * r * np.sqrt(np.maximum(2.0 - r**2, 0.0)),
        sin_phi * r * np.sqrt(np.maximum(2.0 - r**2, 0.0)),
        z
    ]).T
    return np.ascontiguousarray(d)


def compute_riesz_energy(s_order, points):
    # norm is the pairwise i, j euclidean distance between points i and j
    norm = np.sqrt(np.sum((points - points[:, None, :])**2, axis=2))
    if s_order == 0:
        mapped = np.log(1.0 / norm)
    else:
        mapped = norm**(-s_order)
    np.fill_diagonal(mapped, 0.0)
    return np.sum(mapped, axis=1)


if __name__ == '__main__':
    Nphi = 28
    Ntheta = 28
    phis, thetas = np.mgrid[:Nphi, :Ntheta]
    phis = phis.reshape(-1)
    thetas = thetas.reshape(-1)

    if QuadType == 'BasicTrapezoidal':
        dirs = basic_dir(phis, thetas, Nphi, Ntheta)
    elif QuadType == 'OctahedralPbrt':
        dirs = clarberg_pbrt_dir(phis, thetas, Nphi, Ntheta)
    elif QuadType == "ClassicOctahedral":
        dirs = classic_oct_dir(phis, thetas, Nphi, Ntheta)
    elif QuadType == "Healpix":
        hp_order = 3
        hp_nside = healpix.order2nside(hp_order)
        hp_npix = healpix.nside2npix(hp_nside)
        pix = []
        for ipix in range(hp_npix):
            pix.append(healpix.pix2vec(hp_nside, ipix))
        dirs = np.array(pix)
    print(f"{dirs.shape[0]} rays")


    E0 = compute_riesz_energy(0, dirs)
    E1 = compute_riesz_energy(1, dirs)
    EM1 = compute_riesz_energy(-1, dirs)

    fig = plt.figure(layout='constrained', figsize=(10, 4))
    axs = []
    for i, E, label in zip(range(3), [E1, E0, EM1], ['$E_1$', '$E_0$', '$E_{-1}$']):
        if i > 0:
            ax = fig.add_subplot(1, 3, i+1, projection='3d', shareview=axs[0])
        else:
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
        axs.append(ax)
        mappable = ax.scatter(dirs[:, 0], dirs[:, 1], dirs[:, 2], c=E)
        ax.set_title(label)
        fig.colorbar(mappable)

    print(f"E_1: {np.min(E1)}, {np.max(E1)}")
    print(f"E_0: {np.min(E0)}, {np.max(E0)}")
    print(f"E_-1: {np.min(EM1)}, {np.max(EM1)}")
