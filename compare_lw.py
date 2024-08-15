import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from weno4 import weno4
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
wave_ranges = [0.07, 0.07, 0.12]

def centres_to_edges(x):
    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([
        [x[0] - (centres[0] - x[0])],
        centres,
        [x[-1] + (x[-1] - centres[-1])]
    ])

if __name__ == "__main__":
    atmos = xr.open_dataset(atmos_file)
    dex = xr.open_dataset(synth_file)
    ray = xr.open_dataset(ray_file)
    lw_prom = xr.open_dataset(lw_prom_file, group="Prominence")
    lw_fil = xr.open_dataset(lw_fil_file, group="Filament")

    fig, ax = plt.subplots(len(lambda0_idxs), 2, layout="constrained", sharex=True, sharey="row")
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
        ax[plot_idx, 0].pcolormesh(
            slit_pos_edges,
            delta_lambda_edges,
            r_I,
            cmap="inferno"
        )

        lw_slit_cen = np.array(lw_prom.z)
        lw_slit_edges = centres_to_edges(lw_slit_cen) / 1e6
        lw_wave = np.array(lw_prom.wavelength)
        lw_start_idx = np.searchsorted(lw_wave, lambda0-half_delta_lambda)
        lw_end_idx = np.searchsorted(lw_wave, lambda0+half_delta_lambda, side='right') + 1
        lw_delta_lambda_centres = lw_wave[lw_start_idx:lw_end_idx]
        lw_delta_lambda_edges = centres_to_edges(lw_delta_lambda_centres) - lambda0

        lw_I = np.array(lw_prom.I[0, :, 0, lw_start_idx:lw_end_idx])
        ax[plot_idx, 1].pcolormesh(
            lw_slit_edges,
            lw_delta_lambda_edges,
            lw_I.T,
            cmap="inferno"
        )

        ax[plot_idx, 0].set_ylim(-half_delta_lambda, half_delta_lambda)

    ax[plot_idx, 0].set_xlabel("Slit Position [Mm]")
    ax[plot_idx, 1].set_xlabel("Slit Position [Mm]")

    fig.suptitle("Prominence View")
    ax[0, 0].set_title("DexRT")
    ax[0, 1].set_title("Promweaver")


    fig, ax = plt.subplots(len(lambda0_idxs), 2, layout="constrained", sharex=True, sharey="row")
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
        ax[plot_idx, 0].pcolormesh(
            slit_pos_edges,
            delta_lambda_edges,
            r_I,
            cmap="inferno"
        )

        lw_slit_cen = np.array(lw_fil.x)
        lw_slit_edges = centres_to_edges(lw_slit_cen) / 1e6
        lw_wave = np.array(lw_fil.wavelength)
        lw_start_idx = np.searchsorted(lw_wave, lambda0-half_delta_lambda)
        lw_end_idx = np.searchsorted(lw_wave, lambda0+half_delta_lambda, side='right') + 1
        lw_delta_lambda_centres = lw_wave[lw_start_idx:lw_end_idx]
        lw_delta_lambda_edges = centres_to_edges(lw_delta_lambda_centres) - lambda0

        lw_I = np.array(lw_fil.I[:, 0, 0, lw_start_idx:lw_end_idx])
        ax[plot_idx, 1].pcolormesh(
            lw_slit_edges,
            lw_delta_lambda_edges,
            lw_I.T,
            cmap="inferno"
        )

        ax[plot_idx, 0].set_ylim(-half_delta_lambda, half_delta_lambda)
        ax[plot_idx, 0].set_ylabel(r"$\Delta\lambda$ [nm]")

    fig.suptitle("Filament View")
    ax[0, 0].set_title("DexRT")
    ax[0, 1].set_title("Promweaver")
    ax[plot_idx, 0].set_xlabel("Slit Position [Mm]")
    ax[plot_idx, 1].set_xlabel("Slit Position [Mm]")