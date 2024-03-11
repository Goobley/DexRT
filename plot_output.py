import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from skimage.measure import profile_line

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
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(canvas, origin="lower", interpolation="nearest")
        ax[0].set_title("Bilinear Fix + Branching")
        # ax[0].set_title("\"Classical\" RCs")

        start = [511, 0]
        end = [511, 1023]
        ax[0].plot([start[0], end[0]], [start[1], end[1]])
        line1 = profile_line(canvas, start, end)
        start = [0, 511]
        end = [1023, 511]
        ax[0].plot([start[0], end[0]], [start[1], end[1]])
        line2 = profile_line(canvas, start, end)

        start = [511 - int(511/np.sqrt(2)), 511 - int(511/np.sqrt(2))]
        end = [511 + int(511 / np.sqrt(2)), 511 + int(511/np.sqrt(2))]
        ax[0].plot([start[0], end[0]], [start[1], end[1]])
        line3 = profile_line(canvas, start, end)

        ax[1].plot(line1[:, 0])
        ax[1].plot(line2[:, 0])
        ax[1].plot(line3[:, 0])
        ax[1].set_title("Intensity Slices")

        coord = np.arange(1023, dtype=np.float64)
        r_coord = np.abs(coord - 511)
        theory = np.zeros_like(coord)
        PEAK = 4.0
        RADIUS = 40
        theory[:] = PEAK
        theory[r_coord >= RADIUS] = 2.0 * np.arcsin(RADIUS / r_coord[r_coord >= RADIUS]) * PEAK / (2.0 * np.pi)
        ax[1].plot(coord + 1.0, theory, 'k--', label="Theory")
        ax[1].set_yscale('log')
        ax[1].legend()

