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

def centres_to_edges(x):
    centres = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([
        [x[0] - (centres[0] - x[0])],
        centres,
        [x[-1] + (x[-1] - centres[-1])]
    ])

if __name__ == "__main__":
    atmos = xr.open_dataset(atmos_file)
    synth = xr.open_dataset(synth_file)

    fig, ax = plt.subplots(1, 3, layout="constrained", figsize=(9, 3), sharex=True, sharey=True)

    voxel_scale = float(atmos.voxel_scale)
    z_offset = float(atmos.offset_z)
    x_offset = float(atmos.offset_x)
    z_cen = (np.arange(atmos.z.shape[0]) + 0.5) * voxel_scale + z_offset
    z_edges = centres_to_edges(z_cen) / 1e6
    x_cen = (np.arange(atmos.x.shape[0]) + 0.5) * voxel_scale + x_offset
    x_edges = centres_to_edges(x_cen) / 1e6

    mappable = ax[0].pcolormesh(x_edges, z_edges, atmos.temperature, norm=LogNorm(), cmap="inferno", rasterized=True)
    fig.colorbar(mappable, ax=ax[0])
    ax[0].set_ylabel("z [Mm]")
    ax[0].set_xlabel("x [Mm]")
    ax[0].set_title("Temperature [K]")

    mappable = ax[1].pcolormesh(x_edges, z_edges, atmos.pressure, norm=LogNorm(), cmap="magma", rasterized=True)
    fig.colorbar(mappable, ax=ax[1])
    ax[1].set_xlabel("x [Mm]")
    ax[1].set_title("Pressure [Pa]")

    mappable = ax[2].pcolormesh(x_edges, z_edges, synth.ne, norm=LogNorm(), cmap="viridis", rasterized=True)
    fig.colorbar(mappable, ax=ax[2])
    ax[2].set_xlabel("x [Mm]")
    ax[2].set_title("$n_e$ [m$^{-3}$]")

    fig.savefig("jk20200550_params.png", dpi=500)
    fig.savefig("jk20200550_params.pdf", dpi=500)

