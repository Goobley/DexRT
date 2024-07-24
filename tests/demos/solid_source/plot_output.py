import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from dataclasses import dataclass
from tqdm import tqdm

if __name__ == "__main__":
    plt.ion()
    # ds = netCDF4.Dataset("../../../build/circle_test_out_para.nc")
    # ds = netCDF4.Dataset("../../../build/circle_test_out.nc")
    # ds = netCDF4.Dataset("../../../build/circle_test_out_para_inner.nc")
    ds = netCDF4.Dataset("../../../build/circle_test_out_pp.nc")
    figname = "PParaInner_NoMips"
    im = np.array(ds["J"][...])
    im = np.swapaxes(im, 0, 2)

    canvas = np.copy(im)

    fig = plt.figure(layout="constrained", figsize=(10, 6))
    ax = fig.subplot_mosaic(
        """
        AB
        AC
        """)
    ax["A"].imshow(np.tanh(canvas), origin="lower", interpolation="nearest")
    # ax["A"].set_title("Vanilla RC")
    ax["A"].set_title("Parallax Fix Inner")
    # ax["A"].set_title("Bilinear Fix + Branching (No mipmaps)")
    # ax["A"].set_title("\"Classical\" RCs (No mipmaps)")

    centre = 512
    start = [centre, 0]
    end = [centre, 1023]
    # ax["A"].plot([start[0], end[0]], [start[1], end[1]])
    line1 = profile_line(canvas, start, end)
    start = [0, centre]
    end = [1023, centre]
    # ax["A"].plot([start[0], end[0]], [start[1], end[1]])
    line2 = profile_line(canvas, start, end)

    start = [centre - int(centre/np.sqrt(2)), centre - int(centre/np.sqrt(2))]
    end = [centre + int(centre / np.sqrt(2)), centre + int(centre/np.sqrt(2)) - 1.414]
    # ax["A"].plot([start[0], end[0]], [start[1], end[1]])
    line3 = profile_line(canvas, start, end)

    start = [centre - int(centre * np.cos(np.deg2rad(30.0))), centre - int(centre * np.sin(np.deg2rad(30.0)))]
    end = [centre + int(centre *np.cos(np.deg2rad(30.0))), centre + int(centre * np.sin(np.deg2rad(30.0)))]
    # ax["A"].plot([start[0], end[0]], [start[1], end[1]])
    line4 = profile_line(canvas, start, end)

    ax["B"].plot(line1[:, 0])
    ax["B"].plot(line2[:, 0])
    ax["B"].plot(line3[:, 0])
    ax["B"].plot(line4[:, 0])
    ax["B"].set_title("Intensity Slices")

    coord = np.arange(1024, dtype=np.float64)
    r_coord = np.abs(coord - centre)
    theory_solid = np.zeros_like(coord)

    PEAK = 1.0
    tau = 1.0 * 100.0
    source_fn = 8.0
    source_factor = -np.expm1(-tau)
    PEAK = source_factor * source_fn
    RADIUS = 20
    theory_solid[:] = PEAK
    theory_solid[r_coord >= RADIUS] = 2.0 * np.arcsin(RADIUS / r_coord[r_coord >= RADIUS]) * PEAK / (2.0 * np.pi)

    ax["B"].plot(coord, theory_solid, 'k-.', label="Theory Solid")
    ax["B"].set_yscale('log')
    # ax["B"].legend()

    error1 = (line1[:, 0] - theory_solid) / theory_solid
    error2 = (line2[:, 0] - theory_solid) / theory_solid
    error3 = (line3[:, 0] - theory_solid) / theory_solid
    error4 = (line4[:, 0] - theory_solid) / theory_solid
    ax["C"].plot(coord, error1, label="horizontal")
    ax["C"].plot(coord, error2, label="vertical")
    ax["C"].plot(coord, error3, label="45 deg")
    ax["C"].plot(coord, error4, label="30 deg")
    ax["C"].set_yscale("symlog", linthresh=1e-2)
    ax["C"].set_title("Relative Error")
    ax["C"].legend()

    fig.savefig(f"{figname}.png", dpi=300)
