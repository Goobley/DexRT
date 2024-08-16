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
# cfn_file = "JackHighRes/jk20200550_ray_lyb_cfn_all_active.nc"
cfn_file = "JackHighRes/jk20200550_ray_lyb_cfn.nc"
lambda0_idx = 1
# cfn_file = "JackHighRes/jk20200550_ray_caiik_cfn.nc"
# lambda0_idx = 11

def centres_to_edges(x):
    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([
        [x[0] - (centres[0] - x[0])],
        centres,
        [x[-1] + (x[-1] - centres[-1])]
    ])

# NOTE(cmo): This tonemapping function takes colour channel as the first axis,
# so the output needs swapping before display in matplotlib
def tonemap(c, mode='aces', Gamma=2.2, bias=None):
    # http://filmicworlds.com/blog/filmic-tonemapping-operators/
    if mode == 'reinhard':
        c = c / (1.0 + c)
        return c**(1.0 / Gamma)
    elif mode == 'uncharted2':
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30
        W = 11.2
        if bias is None:
            bias = 2.0
        mapper = lambda c: ((c*(A*c+C*B)+D*E)/(c*(A*c+B)+D*F))-E/F
        curr = mapper(c * bias)
        whiteScale = 1.0 / mapper(W)
        return (curr * whiteScale)**(1.0 / Gamma)
    elif mode == 'aces':
        # https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
        # https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
        # https://therealmjp.github.io/posts/sg-series-part-6-step-into-the-baking-lab/
        acesIn = np.ascontiguousarray(
            np.array([
                [0.59719, 0.35458, 0.04823],
                [0.07600, 0.90834, 0.01566],
                [0.02840, 0.13383, 0.83777]
            ]).T
        )
        acesOut = np.ascontiguousarray(
            np.array([
                [1.60475, -0.53108, -0.07367],
                [-0.10208,  1.10813, -0.00605],
                [-0.00327, -0.07276,  1.07602]]
            ).T
        )
        if bias is None:
            bias = 0.8

        def RRTAndODTFit(v):
            # RRT: Reference rendering transform
            # ODT: Output display transform
            a = v * (v + 0.0245786) - 0.000090537
            b = v * (0.983729 * v + 0.4329510) + 0.238081
            return a / b

        def LinearTosRGB(color):
            x = color * 12.92
            y = 1.055 * np.clip(color, 0.0, 1.0)**(1.0 / 2.4) - 0.055

            clr = color
            clr[0] = np.where(color[0] < 0.0031308, x[0], y[0])
            clr[1] = np.where(color[1] < 0.0031308, x[1], y[1])
            clr[2] = np.where(color[2] < 0.0031308, x[2], y[2])
            return clr

        color = np.tensordot(acesIn, c, axes=(0, 0))
        color = RRTAndODTFit(color)
        color = np.tensordot(acesOut, color, axes=(0, 0))
        color = np.clip(color, 0.0, 1.0)
        color = LinearTosRGB(color * bias)
        color = np.clip(color, 0.0, 1.0)
        return color
    elif mode == 'linear':
        c = np.clip(c, 0.0, 1.0)
        return c**(1.0 / Gamma)
    else:
        raise NotImplementedError()

def coco_plot(
    ax,
    arr,
    filt,
    thresh=None,
    log=False,
    max_pre_tonemap=4.0,
    edges=None,
    normalise_channels_individually=True,
    **kwargs
):
    cococfn = np.tensordot(filt, arr, axes=(0, 0))
    # coco_thresh = 1e-7
    # coco_thresh = 1e-5
    # coco_thresh = 1e-4
    if thresh is not None:
        cococfn[cococfn < thresh] = thresh

    if log:
        cococfn[cococfn > 0.0] = np.log10(cococfn[cococfn > 0.0])

    for chan in range(cococfn.shape[0]):
        if normalise_channels_individually:
            max_val = np.nanmax(cococfn[chan])
            min_val = np.nanmin(cococfn[chan])
        else:
            max_val = np.nanmax(cococfn)
            min_val = np.nanmin(cococfn)

        cococfn[chan] = (cococfn[chan] - min_val) / (max_val - min_val) * max_pre_tonemap

    if not "bias" in kwargs:
        kwargs["bias"] = 1.0
    cococfn_tm = tonemap(cococfn, **kwargs)
    if edges is None:
        ax.imshow(np.moveaxis(cococfn_tm, 0, 2), rasterized=True, aspect="auto")
    else:
        ax.pcolormesh(*edges, np.moveaxis(cococfn_tm, 0, 2), rasterized=True)

    return np.moveaxis(cococfn_tm, 0, 2)


if __name__ == "__main__":
    atmos = xr.open_dataset(atmos_file)
    dex = xr.open_dataset(synth_file)
    lambda0 = dex.lambda0[lambda0_idx]
    cfn_ds = xr.open_dataset(cfn_file)
    wave = np.array(cfn_ds.wavelength[...])
    full_cfn = cfn_ds.rays_0_cont_fn[:, :, 1:-1]
    # full_cfn = cfn_ds.rays_0_chi_tau[:, :, 1:-1]
    # full_cfn = np.exp(-cfn_ds.rays_0_tau[:, :, 1:-1]) * cfn_ds.rays_0_tau[:, :, 1:-1]
    voxel_scale = float(atmos.voxel_scale)
    z_offset = float(atmos.offset_z)
    slit_pos_cen = np.array(cfn_ds.ray_start_0[1:-1, 0]) * voxel_scale
    slit_pos_edges = centres_to_edges(slit_pos_cen) / 1e6
    z_pos_cen = np.ascontiguousarray((z_offset + (np.arange(cfn_ds.num_steps_0.shape[0]) + 0.5) * voxel_scale)[::-1])
    z_pos_edges = centres_to_edges(z_pos_cen) / 1e6
    delta_lambda_grid = wave - lambda0
    delta_lambda_edges = centres_to_edges(delta_lambda_grid)

    offsets = [-0.025, 0.0, 0.025]
    means = np.array([lambda0+offset for offset in offsets])
    stds = np.array([0.01, 0.01, 0.01])
    filt = np.exp(-0.5 * ((wave[:, None] - means[None, :]) / stds[None, :])**2)
    filt /= filt.sum(axis=0)

    coco_tau = np.tensordot(filt, cfn_ds.rays_0_tau[:, :, 1:-1], axes=(0, 0))
    tau1_lines = np.zeros((coco_tau.shape[0], coco_tau.shape[2]))
    for chan in range(tau1_lines.shape[0]):
        for col in range(tau1_lines.shape[1]):
            tau1_lines[chan, col] = weno4(
                1.0,
                np.ascontiguousarray(coco_tau[chan, ::-1, col]),
                np.ascontiguousarray(z_pos_cen[::-1]),
            )
    tau1_lines /= 1e6
    del coco_tau

    J_offsets = [-0.025, 0.0, 0.025]
    dex_means = np.array([lambda0+offset for offset in J_offsets])
    dex_stds = np.array([0.01, 0.01, 0.01])
    dex_wave = np.array(dex.wavelength)
    start_idx = np.searchsorted(dex_wave, lambda0 + 3 * offsets[0])
    end_idx = np.searchsorted(dex_wave, lambda0 + 3 * offsets[-1])
    dex_wave = dex_wave[start_idx:end_idx]
    dex_filt = np.exp(-0.5 * ((dex_wave[:, None] - dex_means[None, :]) / dex_stds[None, :])**2)
    dex_filt /= dex_filt.sum(axis=0)

    fig, ax = plt.subplot_mosaic(
        """
        A
        A
        A
        B
        C
        C
        C
        C
        C
        D
        D
        D
        D
        D
        """,
        layout="constrained",
        sharex=True,
        figsize=(4, 10)
    )
    line_emission = np.array(cfn_ds.I_0[:, 1:-1])
    ax["A"].pcolormesh(slit_pos_edges, delta_lambda_edges, line_emission, cmap="inferno", rasterized=True)
    line = line_emission.reshape(101, 1, 2048)
    coco_plot(ax["B"], line, filt, edges=(slit_pos_edges, [0.0, 1.0]))
    coco_cfn = coco_plot(ax["C"], full_cfn, filt, thresh=0.8e-10, log=True, edges=(slit_pos_edges, z_pos_edges))
    coco_J = coco_plot(ax["D"], dex.J[start_idx:end_idx], dex_filt, edges=(slit_pos_edges, z_pos_edges[::-1]), thresh=1e-3)
    ax["C"].plot(slit_pos_cen / 1e6, tau1_lines[0, :], 'r', lw=0.5, alpha=0.8)
    ax["C"].plot(slit_pos_cen / 1e6, tau1_lines[2, :], 'b', lw=0.5, alpha=0.8)
    ax["C"].plot(slit_pos_cen / 1e6, tau1_lines[1, :], 'w', lw=0.5, alpha=0.8)
    ax["C"].plot(slit_pos_cen / 1e6, tau1_lines[1, :], 'k', lw=0.5, alpha=0.8, ls='--')

    ax["A"].set_title("Ly β Spectrum & Formation")
    ax["A"].set_ylabel(r"$\Delta\lambda$ [nm]")
    ax["A"].set_ylim(-0.05, 0.05)
    ax["A"].set_xlim(0.65, 10.99)

    ax["B"].set_ylabel("COCO\nSpectrum")
    ax["B"].tick_params(
        axis="y",
        labelleft=False,
        left=False,
    )
    ax["C"].set_ylabel(r"$z$ [Mm]")
    ax["C"].text(1.11, 27, r"$C_I$", c="#dddddd", verticalalignment="top")
    ax["D"].set_ylabel(r"$z$ [Mm]")
    ax["D"].set_xlabel(r"Slit position [Mm]")
    ax["D"].text(1.11, 27, r"$J$", c="#dddddd", verticalalignment="top")

    fig.savefig("cocoplot_lyb.png", dpi=400)
    fig.savefig("cocoplot_lyb.pdf", dpi=400)

    fig, ax = plt.subplots(2, 3, layout="constrained", sharex=True, sharey=True, figsize=(9, 6))
    ax[0, 0].pcolormesh(slit_pos_edges, z_pos_edges, coco_cfn[:, :, 0], cmap="plasma", rasterized=True)
    ax[0, 1].pcolormesh(slit_pos_edges, z_pos_edges, coco_cfn[:, :, 1], cmap="plasma", rasterized=True)
    ax[0, 2].pcolormesh(slit_pos_edges, z_pos_edges, coco_cfn[:, :, 2], cmap="plasma", rasterized=True)
    ax[1, 0].pcolormesh(slit_pos_edges, z_pos_edges[::-1], coco_J[:, :, 0], cmap="cividis", rasterized=True)
    ax[1, 1].pcolormesh(slit_pos_edges, z_pos_edges[::-1], coco_J[:, :, 1], cmap="cividis", rasterized=True)
    ax[1, 2].pcolormesh(slit_pos_edges, z_pos_edges[::-1], coco_J[:, :, 2], cmap="cividis", rasterized=True)

    ax[0, 0].set_title("Red Channel")
    ax[0, 1].set_title("Green Channel")
    ax[0, 2].set_title("Blue Channel")
    ax[0, 0].set_ylabel("z [Mm]")
    ax[1, 0].set_ylabel("z [Mm]")
    ax[1, 0].set_xlabel("Slit position [Mm]")
    ax[1, 1].set_xlabel("Slit position [Mm]")
    ax[1, 2].set_xlabel("Slit position [Mm]")
    ax[0, 0].text(1.11, 27, r"$C_I$", c="#dddddd", verticalalignment="top")
    ax[1, 0].text(1.11, 27, r"$J$", c="#dddddd", verticalalignment="top")
    fig.savefig("cocoplot_lyb_colourblind_panels.png", dpi=400)
    fig.savefig("cocoplot_lyb_colourblind_panels.pdf", dpi=400)

