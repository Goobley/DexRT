import numpy as np
import netCDF4
import matplotlib.pyplot as plt

PLOT_COMPARISON = False

if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    im = np.array(ds["image"][...])

    J = np.mean(im, axis=2)
    canvas = np.copy(J)


    if PLOT_COMPARISON:
        plt.ion()
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.tanh(canvas), origin="lower", interpolation="nearest")
        ax[0].set_title("Radiance Cascades")
        lw_J = np.load("../LwDexVerification/J_10th.npy")
        ax[1].imshow(np.tanh(lw_J), origin="lower", interpolation="nearest")
        ax[1].set_title("Linear Short Characteristics 10ray/octant (13th order)")
    else:
        plt.ion()
        plt.imshow(canvas, origin="lower", interpolation="nearest")

