import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr
from weno4 import weno4
import astropy.units as u
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

prefix = "jk20200550"
atmos_file = "JackHighRes/jk20200550_dex.nc"
synth_file = "JackHighRes/jk20200550_synth.nc"
ray_file = "JackHighRes/jk20200550_ray.nc"
lw_prom_file = "JackHighRes/LwModels/jk20200550_prom_synth.nc"
lw_fil_file = "JackHighRes/LwModels/jk20200550_fil_synth.nc"
lambda0_idxs = [1, 11, 4]
wave_ranges = [0.045, 0.07, 0.12]
line_names = [r"Ly$\,$β", r"Ca$\,$ɪɪ K", r"H$\,$α"]
prom_name_colours = ["#bbbbbb", "#bbbbbb", "#bbbbbb"]
fil_name_colours = ["#bbbbbb", "#222222", "#222222"]

def centres_to_edges(x):
    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([
        [x[0] - (centres[0] - x[0])],
        centres,
        [x[-1] + (x[-1] - centres[-1])]
    ])

def density_plot(ax, x_flat, y_flat):
    mask = (x_flat > (1e-4 * x_flat.max())) | (y_flat > (1e-4 * y_flat.max()))
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    # https://stackoverflow.com/a/53865762
    bins = 200
    hist, hx_edge, hy_edge = np.histogram2d(x_flat, y_flat, bins=bins, density=True)
    point_colours = RegularGridInterpolator(
        (
            0.5 * (hx_edge[1:] + hx_edge[:-1]),
            0.5 * (hy_edge[1:] + hy_edge[:-1]),
        ),
        hist,
        bounds_error=False,
    )(np.stack([x_flat, y_flat], axis=1))
    point_colours[np.isnan(point_colours)] = 0.0

    idxs = np.argsort(point_colours)
    x_flat = x_flat[idxs]
    y_flat = y_flat[idxs]
    point_colours = point_colours[idxs]
    correlation = pearsonr(x_flat, y_flat)
    max_intens = max(x_flat.max(), y_flat.max())
    min_intens = min(x_flat.min(), y_flat.min())

    ax.scatter(
        x_flat,
        y_flat,
        c=point_colours,
        alpha=0.2,
        rasterized=True,
        norm=LogNorm(),
        cmap="turbo",
        label=f"Pearson R: {correlation.statistic:.3f}"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.plot([min_intens, max_intens], [min_intens, max_intens], '--k')

if __name__ == "__main__":
    atmos = xr.open_dataset(atmos_file)
    dex = xr.open_dataset(synth_file)
    ray = xr.open_dataset(ray_file)
    lw_prom = xr.open_dataset(lw_prom_file, group="Prominence")
    lw_fil = xr.open_dataset(lw_fil_file, group="Filament")

    fig, ax = plt.subplots(len(lambda0_idxs), 3, layout="constrained", figsize=(8, 4))
    for row in range(ax.shape[0]):
        group_y = ax[row, 0].get_shared_y_axes()
        for col in range(1, ax.shape[0], -1):
            ax[row, col].sharex(ax[0, 0])
            ax[row, col].sharey(ax[row, 0])

    for plot_idx, (lambda0_idx, half_delta_lambda) in enumerate(zip(lambda0_idxs, wave_ranges)):
        lambda0 = dex.lambda0[lambda0_idx]

        voxel_scale = float(atmos.voxel_scale)
        z_offset = float(atmos.offset_z)
        x_offset = float(atmos.offset_x)
        slit_pos_cen = np.array(ray.ray_start_0[1:-1, 1]) * voxel_scale + z_offset
        slit_pos_edges = centres_to_edges(slit_pos_cen) / 1e6

        r_wave = np.array(ray.wavelength)
        rstart_idx = np.searchsorted(r_wave, lambda0-half_delta_lambda)
        rend_idx = np.searchsorted(r_wave, lambda0+half_delta_lambda, side='right') + 1
        delta_lambda_centres = r_wave[rstart_idx:rend_idx]
        delta_lambda_edges = centres_to_edges(delta_lambda_centres) - lambda0

        r_I = np.array(ray.I_0[rstart_idx:rend_idx, 1:-1])

        lw_slit_cen = np.array(lw_prom.z)
        lw_slit_edges = centres_to_edges(lw_slit_cen) / 1e6
        lw_wave = np.array(lw_prom.wavelength)
        lw_start_idx = np.searchsorted(lw_wave, lambda0-half_delta_lambda)
        lw_end_idx = np.searchsorted(lw_wave, lambda0+half_delta_lambda, side='right') + 1
        lw_delta_lambda_centres = lw_wave[lw_start_idx:lw_end_idx]
        lw_delta_lambda_edges = centres_to_edges(lw_delta_lambda_centres) - lambda0

        lw_I = np.array(lw_prom.I[0, :, 1, lw_start_idx:lw_end_idx])
        lw_I = (
            lw_I << u.Unit("W / (m2 Hz sr)")
        ).to(
            "kW / (m2 nm sr)",
            equivalencies=u.spectral_density(wav=(lw_wave[lw_start_idx:lw_end_idx] << u.nm))
        ).value

        vmin = min(r_I.min(), lw_I.min())
        vmax = max(r_I.max(), lw_I.max())

        ax[plot_idx, 0].pcolormesh(
            slit_pos_edges,
            delta_lambda_edges,
            r_I,
            cmap="inferno",
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )

        ax[plot_idx, 1].pcolormesh(
            lw_slit_edges,
            lw_delta_lambda_edges,
            lw_I.T,
            cmap="inferno",
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )

        ax[plot_idx, 0].text(
            27,
            half_delta_lambda * 0.9,
            line_names[plot_idx],
            c=prom_name_colours[plot_idx],
            verticalalignment="top",
            horizontalalignment="right"
        )
        ax[plot_idx, 0].set_ylim(-half_delta_lambda, half_delta_lambda)
        ax[plot_idx, 0].set_ylabel(r"$\Delta\lambda$ [nm]")
        ax[plot_idx, 1].tick_params(labelleft=False)

        if plot_idx != len(lambda0_idxs) - 1:
            ax[plot_idx, 0].tick_params(labelbottom=False)
            ax[plot_idx, 1].tick_params(labelbottom=False)

        ax[plot_idx, 2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    ax[plot_idx, 0].set_xlabel("Slit Position [Mm]")
    ax[plot_idx, 1].set_xlabel("Slit Position [Mm]")

    # fig.suptitle("Prominence View")
    ax[0, 0].set_title("DexRT")
    ax[0, 1].set_title("Promweaver")
    ax[0, 2].set_title("Correlation")

    fig.savefig("DexVsPw_jk20200550_prom.png", dpi=300)
    fig.savefig("DexVsPw_jk20200550_prom.pdf", dpi=300)


    fig, ax = plt.subplots(len(lambda0_idxs), 3, layout="constrained", figsize=(8, 4))
    for row in range(ax.shape[0]):
        group_y = ax[row, 0].get_shared_y_axes()
        for col in range(1, ax.shape[0], -1):
            ax[row, col].sharex(ax[0, 0])
            ax[row, col].sharey(ax[row, 0])

    for plot_idx, (lambda0_idx, half_delta_lambda) in enumerate(zip(lambda0_idxs, wave_ranges)):
        lambda0 = dex.lambda0[lambda0_idx]

        voxel_scale = float(atmos.voxel_scale)
        z_offset = float(atmos.offset_z)
        x_offset = float(atmos.offset_x)
        slit_pos_cen = np.array(ray.ray_start_90[1:-1, 0]) * voxel_scale + x_offset
        slit_pos_edges = centres_to_edges(slit_pos_cen) / 1e6

        r_wave = np.array(ray.wavelength)
        rstart_idx = np.searchsorted(r_wave, lambda0-half_delta_lambda)
        rend_idx = np.searchsorted(r_wave, lambda0+half_delta_lambda, side='right') + 1
        delta_lambda_centres = r_wave[rstart_idx:rend_idx]
        delta_lambda_edges = centres_to_edges(delta_lambda_centres) - lambda0

        r_I = np.array(ray.I_90[rstart_idx:rend_idx, 1:-1])

        lw_slit_cen = np.array(lw_fil.x)
        lw_slit_edges = centres_to_edges(lw_slit_cen) / 1e6
        lw_wave = np.array(lw_fil.wavelength)
        lw_start_idx = np.searchsorted(lw_wave, lambda0-half_delta_lambda)
        lw_end_idx = np.searchsorted(lw_wave, lambda0+half_delta_lambda, side='right') + 1
        lw_delta_lambda_centres = lw_wave[lw_start_idx:lw_end_idx]
        lw_delta_lambda_edges = centres_to_edges(lw_delta_lambda_centres) - lambda0

        lw_I = np.array(lw_fil.I[:, 0, 0, lw_start_idx:lw_end_idx])
        lw_I = (
            lw_I << u.Unit("W / (m2 Hz sr)")
        ).to(
            "kW / (m2 nm sr)",
            equivalencies=u.spectral_density(wav=(lw_wave[lw_start_idx:lw_end_idx] << u.nm))
        ).value

        vmin = min(r_I.min(), lw_I.min())
        vmax = max(r_I.max(), lw_I.max())

        ax[plot_idx, 0].pcolormesh(
            slit_pos_edges,
            delta_lambda_edges,
            r_I,
            cmap="inferno",
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )

        ax[plot_idx, 1].pcolormesh(
            lw_slit_edges,
            lw_delta_lambda_edges,
            lw_I.T,
            cmap="inferno",
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )

        density_plot(ax[plot_idx, 2], lw_I.T.flatten(), r_I[:, ::16].flatten())

        ax[plot_idx, 0].text(
            5.4,
            half_delta_lambda * 0.9,
            line_names[plot_idx],
            c=fil_name_colours[plot_idx],
            verticalalignment="top",
            horizontalalignment="right"
        )
        ax[plot_idx, 0].set_ylim(-half_delta_lambda, half_delta_lambda)
        ax[plot_idx, 0].set_ylabel(r"$\Delta\lambda$ [nm]")

        ax[plot_idx, 1].tick_params(labelleft=False)

        if plot_idx != len(lambda0_idxs) - 1:
            ax[plot_idx, 0].tick_params(labelbottom=False)
            ax[plot_idx, 1].tick_params(labelbottom=False)

        ax[plot_idx, 2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)


    # fig.suptitle("Filament View")
    ax[0, 0].set_title("DexRT")
    ax[0, 1].set_title("Promweaver")
    ax[0, 2].set_title("Correlation")
    ax[plot_idx, 0].set_xlabel("Slit Position [Mm]")
    ax[plot_idx, 1].set_xlabel("Slit Position [Mm]")

    fig.savefig("DexVsPw_jk20200550_fil.png", dpi=300)
    fig.savefig("DexVsPw_jk20200550_fil.pdf", dpi=300)