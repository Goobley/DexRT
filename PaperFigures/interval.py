import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon, Arc, Circle, Annulus, CirclePolygon
import matplotlib as mpl
import numpy as np
from shapely import Polygon as PolyShape, LineString
from shapely import intersection
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

try:
    import seaborn as sns
    colors = sns.color_palette("pastel")
except:
    colors = plt.get_cmap("Set2").colors

mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

PROBE_CENTER = (1.3, 0.6)
PROBE_RADIUS = 0.6
PROBE_START_ANGLE = 60.0
PROBE_END_ANGLE = 200.0
PROBE_ANGLE_STEP = 0.05
FAR_LENGTH = 10.0

INT_VIS_RADIUS = 0.3
PROBE_STARTS = [0.0, 0.2, 0.6]
PROBE_ENDS = [FAR_LENGTH, 0.6, 1.0]

LIGHT_GREY = "#888888"
LIGHTER_GREY = "#BBBBBB"

def compress_int_ranges(ints):
    if len(ints) == 0:
        return None

    start_idx = 0
    result = []
    while start_idx < len(ints):
        needle = ints[start_idx]
        idx = start_idx

        while idx < len(ints) and ints[idx] == needle:
            idx += 1

        result.append((needle, start_idx, idx))
        start_idx = idx

    return result

def convert_int_ranges(ranges):
    return list(map(lambda v: (
        v[0],
        v[1] * PROBE_ANGLE_STEP + PROBE_START_ANGLE,
        v[2] * PROBE_ANGLE_STEP + PROBE_START_ANGLE), ranges
    ))

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    prim_colours = ["C0", "C1", "C2"]
    def prim_patches():
        # ellipse = Ellipse((1, 1), 0.5, 0.2, angle=20.0, facecolor=prim_colours[0])
        # ellipse_poly = Polygon(ellipse.get_verts(), facecolor=prim_colours[0])
        circ = CirclePolygon((1, 1), 0.15, resolution=300, facecolor=prim_colours[0])
        rect = Rectangle((0.4, 1.0), 0.35, 0.15, facecolor=prim_colours[1])
        tri = Polygon(
            np.array([
                [0.9, 1.2],
                [1.2, 1.2],
                [1.05, 1.2 + np.sqrt(3) / 2 * 0.3],
            ]),
            facecolor=prim_colours[2]
        )
        return [circ, rect, tri]

    for a in ax:
        for p in prim_patches():
            a.add_patch(p)

    mpl_prims = prim_patches()
    ellipse_shape = PolyShape(np.copy(mpl_prims[0].get_verts()))
    rect_shape = PolyShape(np.copy(mpl_prims[1].get_verts()))
    tri_shape = PolyShape(np.copy(mpl_prims[2].get_verts()))
    # NOTE(cmo): These are "z-sorted" by eye to avoid needing to deal with any of that.
    prims = [ellipse_shape, rect_shape, tri_shape]

    for a in ax:
        a.plot(PROBE_CENTER[0], PROBE_CENTER[1], 'o', c=LIGHT_GREY)
    for a, s, e in zip(ax[1:], PROBE_STARTS[1:], PROBE_ENDS[1:]):
        ann = Annulus(PROBE_CENTER, e, e-s, facecolor=LIGHTER_GREY, edgecolor=LIGHT_GREY, lw=0.5, alpha=0.2)
        a.add_patch(ann)

    for a in ax:
        circ = Circle(PROBE_CENTER, INT_VIS_RADIUS, edgecolor=LIGHT_GREY, lw=0.5, ls='--', fill=False)
        a.add_patch(circ)


    for idx, a in enumerate(ax):
        near_length = PROBE_STARTS[idx]
        far_length = PROBE_ENDS[idx]
        first_int = []
        for theta in np.linspace(
            PROBE_START_ANGLE,
            PROBE_END_ANGLE,
            num=(int((PROBE_END_ANGLE - PROBE_START_ANGLE) / PROBE_ANGLE_STEP) + 1)
        ):
            ray_start = (PROBE_CENTER[0] + np.cos(np.deg2rad(theta)) * near_length, PROBE_CENTER[1] + np.sin(np.deg2rad(theta)) * near_length)
            ray_end = (PROBE_CENTER[0] + np.cos(np.deg2rad(theta)) * far_length, PROBE_CENTER[1] + np.sin(np.deg2rad(theta)) * far_length)
            ray = LineString([ray_start, ray_end])
            intersections = [not intersection(ray, p, grid_size=0).is_empty for p in prims]
            if not any(intersections):
                int_idx = -1
            else:
                int_idx = intersections.index(True)
            first_int.append(int_idx)
        int_ranges = convert_int_ranges(compress_int_ranges(first_int))

        for int_idx, (p_idx, theta1, theta2) in enumerate(int_ranges):
            if theta1 != PROBE_START_ANGLE:
                a.plot(
                    [PROBE_CENTER[0], PROBE_CENTER[0] + np.cos(np.deg2rad(theta1)) * FAR_LENGTH],
                    [PROBE_CENTER[1], PROBE_CENTER[1] + np.sin(np.deg2rad(theta1)) * FAR_LENGTH],
                    c=LIGHT_GREY,
                    lw=0.5,
                    ls="--"
                )

            if p_idx == -1:
                continue

            inner_arc = Arc(
                PROBE_CENTER,
                2 * INT_VIS_RADIUS,
                2 * INT_VIS_RADIUS,
                theta1=theta1,
                theta2=theta2,
                edgecolor=prim_colours[p_idx],
                lw=2.5
            )
            a.add_patch(inner_arc)


    for a in ax:
        a.set_xlim(0.35, 1.35)
        a.set_ylim(0.5, 1.5)
        a.set_axis_off()

    ax[0].text(0.37, 0.51, "a)")
    ax[1].text(0.37, 0.51, "b)")
    ax[2].text(0.37, 0.51, "c)")
    fig.savefig("RadianceInterval.pdf")
    fig.savefig("RadianceInterval.png", dpi=300)