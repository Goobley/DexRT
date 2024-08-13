import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

prefix = "jk20200550"
proxy_file = "jk20200550_hea2015_proxy.nc"
atmos_file = "JackHighRes/jk20200550_dex.nc"
ray_file = "JackHighRes/jk20200550_ray.nc"
lambda0 = 656.4691622298104
xlims = (-0.14, 0.14)

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
    wave_edges = centres_to_edges(wavelengths) - lambda0

    ray_prom = np.array(ray.I_0[wave_start:wave_end, 1:-1])
    ray_prom_tau = np.array(ray.tau_0[wave_start:wave_end, 1:-1])
    ray_fil = np.array(ray[f"I_{ray.mu.shape[0]-1}"][wave_start:wave_end, 1:-1])
    ray_fil_tau = np.array(ray[f"tau_{ray.mu.shape[0]-1}"][wave_start:wave_end, 1:-1])

    voxel_scale = float(atmos.voxel_scale)
    ray_starts_prom = np.array(ray.ray_start_0[1:-1])
    vox_slit_pos_prom = np.sqrt(np.sum((ray_starts_prom - ray_starts_prom[0])**2, axis=1)) - 0.5
    slit_pos_prom = vox_slit_pos_prom * voxel_scale
    ray_starts_fil = np.array(ray[f"ray_start_{ray.mu.shape[0]-1}"][1:-1])
    vox_slit_pos_fil = np.sqrt(np.sum((ray_starts_fil - ray_starts_fil[0])**2, axis=1)) - 0.5
    slit_pos_fil = vox_slit_pos_fil * voxel_scale

    cmap = "magma"
    fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(8, 5.5))

    prom_min = min(proxy.intens_prom.min(), ray_prom.min())
    prom_max = max(proxy.intens_prom.max(), ray_prom.max())

    mappable = ax[0, 0].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_prom) / 1e6,
        proxy.intens_prom,
        cmap=cmap,
        vmin=prom_min,
        vmax=prom_max,
        rasterized=True,
    )
    ax[0, 0].set_xlim(*xlims)
    ax[0, 0].set_title("Hea15 Hα Proxy")
    ax[0, 0].set_ylabel("Slit Position [Mm]")
    ax[0, 0].tick_params(
        labelbottom=False,
    )

    mappable = ax[0, 1].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_prom) / 1e6,
        ray_prom.T,
        cmap=cmap,
        vmin=prom_min,
        vmax=prom_max,
        rasterized=True,
    )
    fig.colorbar(mappable, ax=ax[0, 1])
    ax[0, 1].set_xlim(*xlims)
    ax[0, 1].set_title("DexRT Hα")
    ax[0, 1].tick_params(
        labelleft=False,
        labelbottom=False,
    )

    def density_plot(ax, proxy_flat, ray_flat, proxy_tau=None, ray_tau=None, tau_threshold=1e-6):
        if proxy_tau is not None and ray_tau is not None:
            mask = (proxy_tau > tau_threshold) | (ray_tau > tau_threshold)
        else:
            mask = (proxy_flat > (1e-4 * proxy_flat.max())) | (ray_flat > (1e-4 * ray_flat.max()))
        proxy_flat = proxy_flat[mask]
        ray_flat = ray_flat[mask]
        # https://stackoverflow.com/a/53865762
        bins = 200
        hist, hx_edge, hy_edge = np.histogram2d(proxy_flat, ray_flat, bins=bins, density=True)
        point_colours = RegularGridInterpolator(
            (
                0.5 * (hx_edge[1:] + hx_edge[:-1]),
                0.5 * (hy_edge[1:] + hy_edge[:-1]),
            ),
            hist,
            bounds_error=False,
        )(np.stack([proxy_flat, ray_flat], axis=1))
        point_colours[np.isnan(point_colours)] = 0.0

        idxs = np.argsort(point_colours)
        proxy_flat = proxy_flat[idxs]
        ray_flat = ray_flat[idxs]
        point_colours = point_colours[idxs]
        correlation = pearsonr(proxy_flat, ray_flat)
        max_intens = max(proxy_flat.max(), ray_flat.max())
        min_intens = min(proxy_flat.min(), ray_flat.min())

        ax.scatter(
            proxy_flat,
            ray_flat,
            c=point_colours,
            alpha=0.2,
            rasterized=True,
            norm=LogNorm(),
            cmap="turbo",
            label=f"Pearson R: {correlation.statistic:.3f}"
        )
        ax.legend(frameon=False, loc="lower right")
        ax.plot([min_intens, max_intens], [min_intens, max_intens], '--k')
        ax.set_xlabel("Hea15 Hα proxy")
        ax.set_ylabel("DexRT Hα specific intensity")

    proxy_prom_flat = np.array(proxy.intens_prom).flatten()
    proxy_prom_tau_flat = np.array(proxy.tau_prom).flatten()
    ray_prom_flat = ray_prom.T.flatten()
    ray_prom_tau_flat = ray_prom_tau.T.flatten()
    density_plot(
        ax[0, 2],
        proxy_prom_flat,
        ray_prom_flat,
        proxy_tau=proxy_prom_tau_flat,
        ray_tau=ray_prom_tau_flat,
    )
    ax[0, 2].set_title("Correlation")


    fil_min = min(proxy.intens_fil.min(), ray_fil.min())
    fil_max = max(proxy.intens_fil.max(), ray_fil.max())
    mappable = ax[1, 0].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_fil) / 1e6,
        proxy.intens_fil,
        cmap=cmap,
        vmin=fil_min,
        vmax=fil_max,
        rasterized=True,
    )
    ax[1, 0].set_xlim(*xlims)
    ax[1, 0].set_ylabel("Slit Position [Mm]")
    ax[1, 0].set_xlabel(r"$\Delta\lambda$ [nm]")

    mappable = ax[1, 1].pcolormesh(
        wave_edges,
        centres_to_edges(slit_pos_fil) / 1e6,
        ray_fil.T,
        cmap=cmap,
        vmin=fil_min,
        vmax=fil_max,
        rasterized=True,
    )
    fig.colorbar(mappable, ax=ax[1, 1])
    ax[1, 1].set_xlim(*xlims)
    ax[1, 1].set_xlabel(r"$\Delta\lambda$ [nm]")
    ax[1, 1].tick_params(
        labelleft=False,
    )

    proxy_fil_flat = np.array(proxy.intens_fil).flatten()
    proxy_fil_tau_flat = np.array(proxy.tau_fil).flatten()
    ray_fil_flat = ray_fil.T.flatten()
    ray_fil_tau_flat = ray_fil_tau.T.flatten()
    density_plot(
        ax[1, 2],
        proxy_fil_flat,
        ray_fil_flat,
        proxy_tau=proxy_fil_tau_flat,
        ray_tau=ray_fil_tau_flat,
        tau_threshold=1e-2,
    )
    fig.savefig(f"{prefix}_Hea15Comparison.pdf", dpi=300)
    fig.savefig(f"{prefix}_Hea15Comparison.png", dpi=300)

