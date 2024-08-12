import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

proxy_file = "jk20200550_hea2015_proxy.nc"
atmos_file = "JackHighRes/jk20200550_dex.nc"
ray_file = "JackHighRes/jk20200550_ray.nc"

def centres_to_edges(x):
    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([
        [x[0] - (centres[0] - x[0])],
        centres,
        [x[-1] + (x[-1] - centres[-1])]
    ])

if __name__ == "__main__":
    proxy = xr.open_dataset(proxy_file)
    ray = xr.open_dataset(ray_file)
    atmos = xr.open_dataset(atmos_file)

    # NOTE(cmo): Find wavelength sub-range in ray
    wave_start = np.searchsorted(ray.wavelength, proxy.wavelength[0])
    wave_end = np.searchsorted(ray.wavelength, proxy.wavelength[-1], side="right")+1

    # prom view
    assert float(ray.mu[0, 0]) == -1.0
    # fil view
    assert float(ray.mu[-1, 2]) == 1.0

    wavelengths = np.array(ray.wavelength[wave_start:wave_end])
    wave_edges = centres_to_edges(wavelengths)

    ray_prom = np.array(ray.I_0[wave_start:wave_end, 1:-1])
    ray_fil = np.array(ray[f"I_{ray.mu.shape[0]-1}"][wave_start:wave_end, 1:-1])

    voxel_scale = float(atmos.voxel_scale)
    ray_starts_prom = np.array(ray.ray_start_0[1:-1])
    vox_slit_pos_prom = np.sqrt(np.sum((ray_starts_prom - ray_starts_prom[0])**2, axis=1)) - 0.5
    slit_pos_prom = vox_slit_pos_prom * voxel_scale
    ray_starts_fil = np.array(ray[f"ray_start_{ray.mu.shape[0]-1}"][1:-1])
    vox_slit_pos_fil = np.sqrt(np.sum((ray_starts_fil - ray_starts_fil[0])**2, axis=1)) - 0.5
    slit_pos_fil = vox_slit_pos_fil * voxel_scale

    cmap = "magma"
    fig, ax = plt.subplots(2, 3, constrained_layout=True)

    mappable = ax[0, 0].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_prom) / 1e6,
        proxy.intens_prom,
        cmap=cmap
    )
    fig.colorbar(mappable, ax=ax[0, 0])

    mappable = ax[0, 1].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_prom) / 1e6,
        ray_prom.T,
        cmap=cmap
    )
    fig.colorbar(mappable, ax=ax[0, 1])
    ax[0, 2].plot(
        np.array(proxy.intens_prom).flatten(), ray_prom.T.flatten(),
        'x'
    )
    ax[0, 2].plot([0, 3], [0, 3], c='k')

    mappable = ax[1, 0].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_fil) / 1e6,
        proxy.intens_fil,
        cmap=cmap
    )
    fig.colorbar(mappable, ax=ax[1, 0])

    mappable = ax[1, 1].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_fil) / 1e6,
        ray_fil.T,
        cmap=cmap
    )
    fig.colorbar(mappable, ax=ax[1, 1])
    ax[1, 2].plot(
        np.array(proxy.intens_fil).flatten(), ray_fil.T.flatten(),
        'x'
    )
    ax[1, 2].plot([0, 3], [0, 3], c='k')

