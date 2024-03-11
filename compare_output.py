import numpy as np
import netCDF4
import matplotlib.pyplot as plt

PLOT_COMPARISON = True

if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    im = np.array(ds["image"][...])

    ds_no_mip = netCDF4.Dataset("build/output_no_mip.nc")
    im_no_mip = np.array(ds_no_mip["image"][...])

    J = np.mean(im, axis=2)
    J_no_mip = np.mean(im_no_mip, axis=2)

    plt.ion()
    fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(np.tanh(J), origin="lower", interpolation="nearest")
    # ax[0].set_title("Radiance Cascades (mips)")
    # ax[1].imshow(np.tanh(J_no_mip), origin="lower", interpolation="nearest")
    # ax[1].set_title("Radiance Cascades (no mips)")

    ax[0].imshow(np.tanh(J), origin="lower", interpolation="nearest")
    ax[0].set_title("Radiance Cascades (GT)")
    mappable = ax[1].imshow(np.abs(J - J_no_mip).max(axis=2), origin="lower", interpolation="nearest")
    ax[1].set_title("Absolute Error")
    fig.colorbar(mappable)
