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
        fig = plt.figure(layout="constrained", figsize=(10, 6))
        figname = "NoFix_NoMips"
        ax = fig.subplot_mosaic(
            """
            AB
            AC
            """)
        ax["A"].imshow(canvas, origin="lower", interpolation="nearest")
        # ax["A"].set_title("Bilinear Fix + Branching (mips)")
        ax["A"].set_title("\"Classical\" RCs (No mipmaps)")

        centre = 512
        start = [centre, 0]
        end = [centre, 1023]
        ax["A"].plot([start[0], end[0]], [start[1], end[1]])
        line1 = profile_line(canvas, start, end)
        start = [0, centre]
        end = [1023, centre]
        ax["A"].plot([start[0], end[0]], [start[1], end[1]])
        line2 = profile_line(canvas, start, end)

        start = [centre - int(centre/np.sqrt(2)), centre - int(centre/np.sqrt(2))]
        end = [centre + int(centre / np.sqrt(2)), centre + int(centre/np.sqrt(2)) - 1.414]
        ax["A"].plot([start[0], end[0]], [start[1], end[1]])
        line3 = profile_line(canvas, start, end)

        ax["B"].plot(line1[:, 0])
        ax["B"].plot(line2[:, 0])
        ax["B"].plot(line3[:, 0])
        ax["B"].set_title("Intensity Slices")

        coord = np.arange(1024, dtype=np.float64)
        r_coord = np.abs(coord - centre)
        theory = np.zeros_like(coord)
        PEAK = 4.0
        RADIUS = 40
        theory[:] = PEAK
        theory[r_coord >= RADIUS] = 2.0 * np.arcsin(RADIUS / r_coord[r_coord >= RADIUS]) * PEAK / (2.0 * np.pi)
        ax["B"].plot(coord, theory, 'k--', label="Theory")
        ax["B"].set_yscale('log')
        ax["B"].legend()

        error1 = (line1[:, 0] - theory) / theory
        error2 = (line2[:, 0] - theory) / theory
        error3 = (line3[:, 0] - theory) / theory
        ax["C"].plot(coord, error1)
        ax["C"].plot(coord, error2)
        ax["C"].plot(coord, error3)
        ax["C"].set_yscale("symlog", linthresh=1e-2)
        ax["C"].set_title("Relative Error")

        fig.savefig(f"{figname}.png", dpi=300)


