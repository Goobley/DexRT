import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import yaml
from scipy.ndimage import rotate, convolve1d
from pathlib import Path
from weno4 import weno4
import astropy.constants as const

atom_name = "Ca II"
lambda0 = 393.47771342231175
delta_lambda = 0.017
delta_lambda = 0.082
# outdir = Path("SnowKHIRay")
# outdir = Path("Valeriia_a0500")
outdir = Path("JackHighRes/Frames")
outdir.mkdir(parents=True, exist_ok=True)

# prefix = "build"
prefix = "JackHighRes"
config_file = f"jk20200550_ray.yaml"

# atom_name = "Ca II"
# lambda0 = 854.4437912696352
# # delta_lambda = 0.017
# delta_lambda = 0.11


atom_name = "H I"
# lambda0 = 121.56841096386113
# delta_lambda = 0.07
lambda0 = 102.57334047695103
delta_lambda = 0.07
# lambda0 = 656.4691622298104
# delta_lambda = 0.17

line_name = f"{atom_name} {lambda0:.2f} nm"
line_name_save = line_name.replace(" ", "_")

DO_SPECTRAL_CONVOLVE = False
# NOTE(cmo): Sumer
# spectral_sigma = 0.0043
# NOTE(cmo): SPICE ?
spectral_sigma = 0.009
# NOTE(cmo): ViSP PSF?
# spectral_sigma = 1.75e-3
DO_SPECTRAL_DOWNSAMPLE = False
# NOTE(cmo): Sumer
# spectral_bin = 0.0043
# NOTE(cmo): SPICE
spectral_bin = 0.009
# NOTE(cmo): ViSP
# spectral_bin = 1.75e-3
DO_SPATIAL_BINNING = False
# NOTE(cmo): Assuming EUI pixel size
spatial_bin = 120e3
# NOTE(cmo): ViSP - 51 km...
# spatial_bin = np.sin(np.deg2rad(0.07 / 3600)) * const.au.value

def centres_to_edges(x):
    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([
        [x[0] - (centres[0] - x[0])],
        centres,
        [x[-1] + (x[-1] - centres[-1])]
    ])

if __name__ == "__main__":
    with open(f"{prefix}/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    ray_data = netCDF4.Dataset(f"{prefix}/" + config["ray_output_path"], "r")
    dexrt_config_path = f"{prefix}/" + config["dexrt_config_path"]

    with open(dexrt_config_path, "r") as f:
        dexrt_config = yaml.load(f, Loader=yaml.Loader)
    atmos = netCDF4.Dataset(f"{prefix}/" + dexrt_config["atmos_path"], "r")
    dex_output = netCDF4.Dataset(f"{prefix}/" + dexrt_config["output_path"], "r")


    voxel_scale = atmos["voxel_scale"][...] / 1e6

    wavelength = ray_data["wavelength"][...]
    start_la = max(np.searchsorted(wavelength, lambda0 - delta_lambda) - 1, 0)
    end_la = min(np.searchsorted(wavelength, lambda0 + delta_lambda) + 1, wavelength.shape[0])

    temperature = atmos["temperature"][...]

    fixed_min_x, fixed_max_x, fixed_min_z, fixed_max_z = np.inf, -np.inf, np.inf, -np.inf
    min_I = 0.0
    max_I = 0.0
    for m in range(ray_data["mu"].shape[0]):
        starts = ray_data[f"ray_start_{m}"][...]
        fixed_min_x = min(np.min(starts[:, 0]), fixed_min_x)
        fixed_max_x = max(np.max(starts[:, 0]), fixed_max_x)
        fixed_min_z = min(np.min(starts[:, 1]), fixed_min_z)
        fixed_max_z = max(np.max(starts[:, 1]), fixed_max_z)
        I = ray_data[f"I_{m}"][start_la:end_la]
        max_I = max(np.max(I), max_I)

    # temperature = dex_output["image"][np.searchsorted(wavelength, lambda0)]

    figsize=(12, 9)
    cmap = "magma"
    fixed_I_scale = False

    for mu_idx in tqdm(range(ray_data["mu"].shape[0])):

        I = ray_data[f"I_{mu_idx}"][start_la:end_la]
        tau = ray_data[f"tau_{mu_idx}"][start_la:end_la].T

        # plt.ion()
        # fig, ax =  plt.subplots(1, 3, constrained_layout=True)
        fig, ax = plt.subplot_mosaic("AB\nCC", constrained_layout=False, figsize=figsize)
        ray_starts = np.asarray(ray_data[f"ray_start_{mu_idx}"][...])
        vox_slit_pos = np.sqrt(np.sum((ray_starts - ray_starts[0])**2, axis=1)) - 0.5
        slit_pos = vox_slit_pos * voxel_scale

        intensity = I.T
        if DO_SPECTRAL_CONVOLVE:
            waves = wavelength - lambda0
            kernel = 1.0 / (spectral_sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * waves**2 / spectral_sigma**2) * (waves[1] - waves[0])
            intensity = convolve1d(intensity, kernel, axis=1, mode="constant")
        if DO_SPECTRAL_DOWNSAMPLE:
            wave_interp_start = ((wavelength[start_la] + spectral_bin) // spectral_bin) * spectral_bin
            wave_interp_end = (wavelength[end_la] // spectral_bin) * spectral_bin
            wave_interp_steps = int((wave_interp_end - wave_interp_start) / (spectral_bin / 10))
            wave_interp = np.linspace(wave_interp_start, wave_interp_end, wave_interp_steps)
            wave_grid = wave_interp[::10]
            int_interp = np.zeros((intensity.shape[0], wave_interp.shape[0]))
            tau_interp = np.zeros((intensity.shape[0], wave_interp.shape[0]))
            for k in range(intensity.shape[0]):
                int_interp[k, :] = weno4(wave_interp, wavelength[start_la:end_la], intensity[k, :])
                tau_interp[k, :] = weno4(wave_interp, wavelength[start_la:end_la], tau[k, :])
            intensity = np.mean(int_interp.reshape(int_interp.shape[0], -1, 10), axis=-1)
            tau = np.mean(tau_interp.reshape(tau_interp.shape[0], -1, 10), axis=-1)
        else:
            wave_grid = wavelength[start_la:end_la]
        if DO_SPATIAL_BINNING:
            bin = spatial_bin / 1e6
            slit_pos_start = ((slit_pos[0] + bin) // bin) * bin
            slit_pos_end = (slit_pos[-1] // bin) * bin
            slit_pos_steps = int((slit_pos_end - slit_pos_start) / (bin / 100))
            slit_interp = np.linspace(slit_pos_start, slit_pos_end, slit_pos_steps)
            slit_grid = slit_interp[::100]
            int_interp = np.zeros((slit_interp.shape[0], intensity.shape[1]))
            tau_interp = np.zeros((slit_interp.shape[0], intensity.shape[1]))
            for la in range(int_interp.shape[1]):
                int_interp[:, la] = weno4(slit_interp, slit_pos, intensity[:, la])
                tau_interp[:, la] = weno4(slit_interp, slit_pos, tau[:, la])
            intensity = np.mean(int_interp.reshape(-1, 100, int_interp.shape[1]), axis=1)
            tau = np.mean(tau_interp.reshape(-1, 100, tau_interp.shape[1]), axis=1)
        else:
            slit_grid = slit_pos

        iax = ax["A"]
        vmin, vmax = None, None
        if fixed_I_scale:
            vmin = min_I
            vmax = max_I
        mappable = iax.pcolormesh(
            centres_to_edges(wave_grid - lambda0),
            centres_to_edges(slit_grid),
            intensity,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        iax.set_xlim(-delta_lambda, delta_lambda)
        iax.set_xlabel(r"$\Delta\lambda$ [nm]")
        iax.set_ylabel(r"Slit Position [Mm]")
        iax.set_title(f"{line_name} Spectrogram")
        iax.tick_params("both", direction="in")
        iax.tick_params("x", labelrotation=30.0)
        fig.colorbar(mappable, ax=iax)

        angle = 1.5 * np.pi - np.sign(ray_data["mu"][mu_idx, 0]) * np.arccos(ray_data["mu"][mu_idx, 2])
        rot_temp = rotate(temperature, angle=np.rad2deg(angle), reshape=True, mode="nearest")
        # rot_temp = temperature.T[:, ::-1]
        rotax = ax["B"]
        rotax.sharey(iax)
        mappable = rotax.imshow(
            rot_temp,
            norm=LogNorm(),
            origin="lower",
            extent=[0.0, rot_temp.shape[1] * voxel_scale, 0.0, rot_temp.shape[0] * voxel_scale],
            aspect="auto",
            cmap=cmap,
        )
        rotax.set_xlabel("Inclined pos [Mm]")
        rotax.set_title("Temperature [K]")
        rotax.tick_params("both", direction="in", labelleft=False)

        fig.colorbar(mappable, ax=rotax)

        viewrayax = ax["C"]
        offset_x = atmos["offset_x"][...] / 1e6
        offset_z = atmos["offset_z"][...] / 1e6
        viewrayax.imshow(
            temperature,
            extent=[
                offset_x,
                temperature.shape[1] * voxel_scale + offset_x,
                offset_z,
                temperature.shape[0] * voxel_scale + offset_z,
            ],
            origin="lower",
            norm=LogNorm(),
            cmap=cmap,
        )
        viewrayax.set_ylabel("z [Mm]")
        viewrayax.set_xlabel("x [Mm]")
        viewrayax.set_xlim(fixed_min_x * voxel_scale + offset_x, fixed_max_x * voxel_scale + offset_x)
        viewrayax.set_ylim(fixed_min_z * voxel_scale + offset_z, fixed_max_z * voxel_scale + offset_z)

        viewrayax.arrow(
            ray_starts[0, 0] * voxel_scale + offset_x,
            ray_starts[0, 1] * voxel_scale + offset_z,
            (ray_starts[-1, 0] - ray_starts[0, 0]) * voxel_scale,
            (ray_starts[-1, 1] - ray_starts[0, 1]) * voxel_scale,
            length_includes_head=True,
            head_width=0.2,
            color="C0",
        )
        arrow_x = ray_starts[0, 0] * voxel_scale + offset_x
        arrow_y = ray_starts[0, 1] * voxel_scale + offset_z
        arrow_dx = (ray_starts[-1, 0] - ray_starts[0, 0]) * voxel_scale
        arrow_dy = (ray_starts[-1, 1] - ray_starts[0, 1]) * voxel_scale
        viewrayax.arrow(
            arrow_x,
            arrow_y,
            arrow_dx,
            arrow_dy,
            length_includes_head=True,
            head_width=0.2,
            color="C0",
        )
        view_arrow_length = 0.4 * np.sqrt(arrow_dx**2 + arrow_dy**2)
        viewrayax.arrow(
            arrow_x + 0.5 * arrow_dx,
            arrow_y + 0.5 * arrow_dy,
            -view_arrow_length * ray_data["mu"][mu_idx, 0],
            -view_arrow_length * ray_data["mu"][mu_idx, 2],
            length_includes_head=True,
            head_width=0.2,
            color="C1"
        )
        print_angle = np.rad2deg(np.sign(ray_data["mu"][mu_idx, 0]) * np.arccos(ray_data["mu"][mu_idx, 2]))
        viewrayax.set_title(f"Imaging plane @ {print_angle:.1f} °")
        viewrayax.tick_params("both", direction="in")

        tau_fig, tax = plt.subplots(1, 1)
        mappable = tax.pcolormesh(
            centres_to_edges(wave_grid - lambda0),
            centres_to_edges(slit_grid),
            tau,
            cmap=cmap,
        )
        tax.set_xlim(-delta_lambda, delta_lambda)
        tax.set_xlabel(r"$\Delta\lambda$ [nm]")
        tax.set_ylabel(r"Slit Position [Mm]")
        tax.set_title(f"{line_name} Optical Depth")
        tax.grid()
        tax.tick_params("both", direction="in")
        tax.tick_params("x", labelrotation=30.0)
        tau_fig.colorbar(mappable, ax=tax)

        fig.savefig(outdir / f"{line_name_save}_{mu_idx:03d}.png", dpi=300)
        tau_fig.savefig(outdir / f"{line_name_save}_tau_{mu_idx:03d}.png", dpi=300)
        plt.close(fig)
        plt.close(tau_fig)

# ffmpeg -framerate 12 -i SnowKHIRay/%03d.png -pix_fmt yuv420p -vcodec libx264 -movflags +faststart -vf "format=yuv420p,scale=1920:1080" -crf 10 test.mp4