import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import yaml
from scipy.ndimage import rotate
from pathlib import Path

atom_name = "Ca II"
lambda0 = 393.47771342231175
delta_lambda = 0.017
outdir = Path("SnowKHIRay")
outdir.mkdir(parents=True, exist_ok=True)

# lambda0 = 854.4437912696352

line_name = f"Ca II {lambda0:.2f} nm"
line_name_save = line_name.replace(" ", "_")

def centres_to_edges(x):
    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([
        [x[0] - (centres[0] - x[0])],
        centres,
        [x[-1] + (x[-1] - centres[-1])]
    ])

if __name__ == "__main__":
    with open("build/dexrt_ray_91.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    atmos = netCDF4.Dataset("build/" + config["atmos_path"], "r")
    dex_output = netCDF4.Dataset("build/" + config["dex_output_path"], "r")
    ray_data = netCDF4.Dataset("build/ray_output.nc", "r")

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
        tau = ray_data[f"tau_{mu_idx}"][start_la:end_la]

        # plt.ion()
        # fig, ax =  plt.subplots(1, 3, constrained_layout=True)
        fig, ax = plt.subplot_mosaic("AB\nCC", constrained_layout=False, figsize=figsize)
        ray_starts = np.asarray(ray_data[f"ray_start_{mu_idx}"][...])
        vox_slit_pos = np.sqrt(np.sum((ray_starts - ray_starts[0])**2, axis=1)) - 0.5
        slit_pos = vox_slit_pos * voxel_scale

        iax = ax["A"]
        vmin, vmax = None, None
        if fixed_I_scale:
            vmin = min_I
            vmax = max_I
        mappable = iax.pcolormesh(
            centres_to_edges(wavelength[start_la :end_la] - lambda0),
            centres_to_edges(slit_pos),
            I.T,
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
            centres_to_edges(wavelength[start_la :end_la] - lambda0),
            centres_to_edges(slit_pos),
            tau.T,
            cmap=cmap,
        )
        tax.set_xlim(-delta_lambda, delta_lambda)
        tax.set_xlabel(r"$\Delta\lambda$ [nm]")
        tax.set_ylabel(r"Slit Position [Mm]")
        tax.set_title(f"{line_name} Spectrogram")
        tax.tick_params("both", direction="in")
        tax.tick_params("x", labelrotation=30.0)
        tau_fig.colorbar(mappable, ax=tax)

        fig.savefig(outdir / f"{line_name_save}_{mu_idx:03d}.png", dpi=300)
        tau_fig.savefig(outdir / f"{line_name_save}_tau_{mu_idx:03d}.png", dpi=300)
        plt.close(fig)
        plt.close(tau_fig)

# ffmpeg -framerate 12 -i SnowKHIRay/%03d.png -pix_fmt yuv420p -vcodec libx264 -movflags +faststart -vf "format=yuv420p,scale=1920:1080" -crf 10 test.mp4