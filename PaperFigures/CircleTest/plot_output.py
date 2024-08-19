import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib as mpl

try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

try:
    import seaborn as sns
    colors = sns.color_palette("muted")
except:
    colors = plt.get_cmap("Set2").colors

mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

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
            # y = 1.055 * np.clip(color, 0.0, 1.0)**(1.0 / Gamma) - 0.055
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


if __name__ == "__main__":
    def plot_grid_for_im(ax, im, image_label=None, legend=True):
        prebias = 1.0
        imm = prebias * np.swapaxes(im, 0, 2)
        im_tm = np.swapaxes(tonemap(imm, bias=1.0), 0, 2)
        ax["A"].imshow(im_tm, origin="lower", rasterized=True)
        ax["A"].tick_params(axis="both", labelleft=False, labelbottom=False, left=False, bottom=False)

        centre = 512
        start = [centre, centre]
        end = [centre, 1023]
        line1 = profile_line(im, start, end, order=0)
        start = [centre, centre]
        end = [1023, centre]
        line2 = profile_line(im, start, end, order=0)

        start = [centre, centre]
        end = [centre + 361.33, centre + 361.33]
        line3 = profile_line(im, start, end, order=0)

        # start = [centre - int(centre * np.cos(np.deg2rad(30.0))), centre - int(centre * np.sin(np.deg2rad(30.0)))]
        start = [centre, centre]
        end = [centre + 511*np.cos(np.deg2rad(30.0)), centre + 511*np.sin(np.deg2rad(30.0))]
        line4 = profile_line(im, start, end)

        ax["B"].plot(line1[:, 0], label=r"0$\degree$")
        ax["B"].plot(line2[:, 0], label=r"90$\degree$")
        ax["B"].plot(line3[:, 0], label=r"45$\degree$")
        ax["B"].plot(line4[:, 0], label=r"30$\degree$", c="C4")
        # ax["B"].set_title("Intensity Slices")

        # coord = np.arange(512, dtype=np.float64) + 0.5
        coord = np.arange(512, dtype=np.float64)
        # r_coord = np.abs(coord - centre)
        r_coord = coord
        theory_solid = np.zeros_like(coord)

        PEAK = 1.0
        tau = 1.0 * 100.0
        source_fn = 8.0
        source_factor = -np.expm1(-tau)
        PEAK = source_factor * source_fn
        RADIUS = 20
        theory_solid[:] = PEAK
        theory_solid[r_coord >= RADIUS] = 2.0 * np.arcsin(RADIUS / r_coord[r_coord >= RADIUS]) * PEAK / (2.0 * np.pi)

        ax["B"].plot(coord, theory_solid, 'k-.', label="Theory")
        ax["B"].set_ylabel("Intensity")
        ax["B"].set_yscale('log')
        if legend:
            ax["B"].legend(frameon=False, labelspacing=0.25)
        # ax["B"].legend()

        error1 = (line1[:, 0] - theory_solid) / theory_solid
        error2 = (line2[:, 0] - theory_solid) / theory_solid
        error3 = (line3[:, 0] - theory_solid) / theory_solid
        error4 = (line4[:, 0] - theory_solid) / theory_solid
        ax["C"].plot(coord, error1, label="horizontal")
        ax["C"].plot(coord, error2, label="vertical")
        ax["C"].plot(coord, error3, label="45 deg")
        ax["C"].plot(coord, error4, label="30 deg", c="C4")
        ax["C"].set_yscale("symlog", linthresh=1e-1)
        ax["C"].set_yticks([1.0, 0.1, 0.01, 0.0])
        ax["C"].set_ylim(-5e-3, 1.0)
        # ax["C"].set_title("Relative Error")
        ax["C"].set_ylabel("Relative Error")
        # ax["C"].legend()
        ax["B"].tick_params(axis="x", labelbottom=False)
        ax["C"].tick_params(axis="x", labelbottom=False)

        if image_label is not None:
            ax["A"].text(40, 984, image_label, verticalalignment="top", c="#eeeeee")


    base = netCDF4.Dataset("circle_test_out.nc")
    bilin = netCDF4.Dataset("circle_test_out_bilin.nc")
    im = np.array(base["J"][...])
    im = np.swapaxes(im, 0, 2)
    im_bilin = np.array(bilin["J"][...])
    im_bilin = np.swapaxes(im_bilin, 0, 2)


    fig = plt.figure(layout="tight", figsize=(10, 6))
    ax = fig.subplot_mosaic(
        """
        AABBB
        AACCC
        DDEEE
        DDFFF
        """
    )
    ax["C"].sharex(ax["B"])
    ax["E"].sharex(ax["B"])
    ax["F"].sharex(ax["B"])
    ax["A"].sharex(ax["D"])
    ax["A"].sharey(ax["D"])
    plot_grid_for_im(ax, im, image_label="Radiance Cascades")
    ax2 = {
        "A": ax["D"],
        "B": ax["E"],
        "C": ax["F"],
    }
    plot_grid_for_im(ax2, im_bilin, image_label="Bilinear Fix", legend=False)
    ax["F"].tick_params(axis="x", labelbottom=True, bottom=True)
    ax["F"].set_xlabel("Pixels From Centre")

    fig.savefig("CircleTest.png", dpi=300)
    fig.savefig("CircleTest.pdf", dpi=300)
