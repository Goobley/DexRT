import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Arc
import matplotlib as mpl
import numpy as np
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap("Set2").colors)

SOURCE_OFFSET = 0.0
CENTRED = True
ADD_LABELS = True
NEAR_ANGLE = True
FAR_ANGLE = True
B_PLANE_Y = 0.5

# NOTE(cmo): Getting close to small angle approx
# BLOCKER_PERP_DIST = 0.05
# SOURCE_SIZE = 0.05
# A_PLANE_Y = 3.7

# NOTE(cmo): Bigger angle - normal figure
BLOCKER_PERP_DIST = 1.0
SOURCE_SIZE = 0.8
A_PLANE_Y = 2.5

def gamma_centred(w, d):
    v = -w/2
    sin_gamma = w * d / np.sqrt((d**2 + v**2)**2 + (d**2 + v**2)*w*(2*v+w))
    return np.arcsin(sin_gamma)

def gamma_offset(w, d, v):
    sin_gamma = (w * d) / np.sqrt((d**2 + v**2) * (d**2 + (v+w)**2))
    return np.arcsin(sin_gamma)

if __name__ == "__main__":
    print(f"BLOCKER_PERP_DIST: {BLOCKER_PERP_DIST}")
    print(f"SOURCE_SIZE: {SOURCE_SIZE}")
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.set_aspect("equal")
    ax.set_axis_off()
    CANVAS_SIZE = (4.0, 4.0)
    ax.set_xlim(0.0, CANVAS_SIZE[0])
    ax.set_ylim(0.0, CANVAS_SIZE[1]+0.01)
    blocker_y = CANVAS_SIZE[1] - BLOCKER_PERP_DIST
    if CENTRED:
        SOURCE_OFFSET = -0.5 * SOURCE_SIZE

    # draw source
    source_start = 0.5 * CANVAS_SIZE[0] + SOURCE_OFFSET
    source_end = source_start + SOURCE_SIZE
    plt.plot([source_start, source_end], [CANVAS_SIZE[1], CANVAS_SIZE[1]], lw=3, c="C5")
    # draw blocker
    blocker_start = 0.0
    blocker_end = 0.5 * CANVAS_SIZE[0]
    plt.plot([blocker_start, blocker_end], [blocker_y, blocker_y], lw=4, c='k', solid_capstyle="butt")

    line1 = np.array([blocker_end, blocker_y]) - np.array([source_start, CANVAS_SIZE[1]])
    line1_dir = line1 / np.linalg.norm(line1)
    line2 = np.array([blocker_end, blocker_y]) - np.array([source_end, CANVAS_SIZE[1]])
    line2_dir = line2 / np.linalg.norm(line2)
    line_angle = np.arccos((line1 @ line2) / (np.linalg.norm(line1) * np.linalg.norm(line2)))
    pred_angle = gamma_offset(SOURCE_SIZE, BLOCKER_PERP_DIST, SOURCE_OFFSET)

    t1_max = (0.0 - CANVAS_SIZE[1]) / line1_dir[1]
    t2_max = (0.0 - CANVAS_SIZE[1]) / line2_dir[1]

    my_grey = "#777777"
    plt.plot([source_start, source_start + t1_max * line1_dir[0]], [CANVAS_SIZE[1], CANVAS_SIZE[1] + t1_max * line1_dir[1]], '--', c=my_grey)
    plt.plot([source_end, source_end + t2_max * line2_dir[0]], [CANVAS_SIZE[1], CANVAS_SIZE[1] + t2_max * line2_dir[1]], '--', c=my_grey)

    resolution = 400
    x_grid = np.linspace(0.0, CANVAS_SIZE[0], resolution+1)
    x_grid = 0.5 * (x_grid[1:] + x_grid[:-1])
    y_grid = np.linspace(0.0, CANVAS_SIZE[1], resolution+1)
    y_grid = 0.5 * (y_grid[1:] + y_grid[:-1])
    xx, yy  = np.meshgrid(x_grid, y_grid)
    im = np.ones((resolution, resolution))
    dark = 0.0
    full_light = 1.0
    im[yy < blocker_y] = dark
    t1 = (yy - CANVAS_SIZE[1]) / line1_dir[1]
    t2 = (yy - CANVAS_SIZE[1]) / line2_dir[1]
    im[(yy < blocker_y) & (xx > (source_start + t1 * line1_dir[0]))] = full_light
    mask = (yy < blocker_y) & (xx <= (source_start + t1 * line1_dir[0])) & (xx >= (source_end + t2 * line2_dir[0]))
    im[mask] = ((xx - (source_end + t2 * line2_dir[0])) / ((source_start + t1 * line1_dir[0]) - (source_end + t2 * line2_dir[0])))[mask]**1.4
    plt.imshow(im, extent=[0.0, CANVAS_SIZE[0], CANVAS_SIZE[1], 0.0], cmap='Purples_r', vmax=1.2, rasterized=True)

    if ADD_LABELS:
        plt.text(blocker_start + 0.05, blocker_y - 0.02, "Opaque Blocker", c=my_grey, verticalalignment="top", fontsize="large")
        plt.text(blocker_start + 0.2, 2.0, "Shadow", c=my_grey, verticalalignment="top", fontsize="large")
        plt.text(source_start - 0.02, CANVAS_SIZE[1], "Light source", c=my_grey, verticalalignment="top", horizontalalignment="right", fontsize="large")
        x1 = source_start + t1_max * line1_dir[0]
        x2 = source_end + t2_max * line2_dir[0]
        plt.text(0.5 * (x1 + x2), 0.1, "Penumbra", c="k", horizontalalignment="center", fontsize="large")

    angle_lw = 2.0
    if NEAR_ANGLE:
        t1 = (A_PLANE_Y - CANVAS_SIZE[1]) / line1_dir[1]
        t2 = (A_PLANE_Y - CANVAS_SIZE[1]) / line2_dir[1]
        plt.plot(
            [source_start + t1 * line1_dir[0], source_end + t2 * line2_dir[0]],
            [A_PLANE_Y, A_PLANE_Y],
            c='C2',
        )
        plt.plot(
            [source_start + t1 * line1_dir[0], source_start],
            [A_PLANE_Y, CANVAS_SIZE[1]],
            c="C4",
            lw=angle_lw,
        )
        plt.plot(
            [source_start + t1 * line1_dir[0], source_end],
            [A_PLANE_Y, CANVAS_SIZE[1]],
            c="C4",
            lw=angle_lw,
        )
        plt.plot(
            [source_start + t1 * line1_dir[0], source_end + t2 * line2_dir[0]],
            [A_PLANE_Y, A_PLANE_Y],
            'o',
            c='C3',
        )
        arc_centre = (source_start + t1 * line1_dir[0], A_PLANE_Y)
        angle0 = np.rad2deg(np.arctan2(CANVAS_SIZE[1] - arc_centre[1], source_end - arc_centre[0]))
        angle1 = np.rad2deg(np.arctan2(CANVAS_SIZE[1] - arc_centre[1], source_start - arc_centre[0]))
        arc_diam = 0.6
        angle_arc = Arc(arc_centre, arc_diam, arc_diam, theta1=angle0, theta2=angle1, fill=False, edgecolor="C4", lw=angle_lw)
        ax.add_patch(angle_arc)

        a_length = source_start + t1 * line1_dir[0] - (source_end + t2 * line2_dir[0])
        alpha = angle1 - angle0
        print(f"A: {a_length:.3f}")
        print(f"alpha: {alpha:.3f} deg")

        if ADD_LABELS:
            plt.text(0.5 * (source_start + t1 * line1_dir[0] + source_end + t2 * line2_dir[0]), A_PLANE_Y - 0.04, "A", verticalalignment="top", horizontalalignment="center", fontsize="large", c="k")
            avg_angle = np.deg2rad(0.5 * (angle0 + angle1))
            angle_dist = 0.5 * arc_diam + 0.02
            plt.text(arc_centre[0] + np.cos(avg_angle) * angle_dist, arc_centre[1] + np.sin(avg_angle) * angle_dist, r"$\alpha$", horizontalalignment="center")


    if FAR_ANGLE:
        t1 = (B_PLANE_Y - CANVAS_SIZE[1]) / line1_dir[1]
        t2 = (B_PLANE_Y - CANVAS_SIZE[1]) / line2_dir[1]
        plt.plot(
            [source_start + t1 * line1_dir[0], source_end + t2 * line2_dir[0]],
            [B_PLANE_Y, B_PLANE_Y],
            c='C2',
        )
        plt.plot(
            [source_start + t1 * line1_dir[0], source_start],
            [B_PLANE_Y, CANVAS_SIZE[1]],
            ls='--',
            lw=angle_lw,
            c="C6"
        )
        plt.plot(
            [source_start + t1 * line1_dir[0], source_end],
            [B_PLANE_Y, CANVAS_SIZE[1]],
            ls='--',
            lw=angle_lw,
            c="C6"
        )
        plt.plot(
            [source_start + t1 * line1_dir[0], source_end + t2 * line2_dir[0]],
            [B_PLANE_Y, B_PLANE_Y],
            'o',
            c='C3',
        )
        arc_centre = (source_start + t1 * line1_dir[0], B_PLANE_Y)
        angle0 = np.rad2deg(np.arctan2(CANVAS_SIZE[1] - arc_centre[1], source_end - arc_centre[0]))
        angle1 = np.rad2deg(np.arctan2(CANVAS_SIZE[1] - arc_centre[1], source_start - arc_centre[0]))
        arc_diam = 0.8
        angle_arc = Arc(arc_centre, arc_diam, arc_diam, theta1=angle0, theta2=angle1, fill=False, edgecolor="C6", lw=angle_lw)
        ax.add_patch(angle_arc)

        b_length = source_start + t1 * line1_dir[0] - (source_end + t2 * line2_dir[0])
        beta = angle1 - angle0
        print(f"B: {b_length:.3f}")
        print(f"beta: {beta:.3f} deg")

        if ADD_LABELS:
            plt.text(0.5 * (source_start + t1 * line1_dir[0] + source_end + t2 * line2_dir[0]), B_PLANE_Y - 0.04, "B", verticalalignment="top", horizontalalignment="center", fontsize="large", c="k")

            avg_angle = np.deg2rad(0.5 * (angle0 + angle1))
            angle_dist = 0.5 * arc_diam + 0.02
            plt.text(arc_centre[0] + np.cos(avg_angle) * angle_dist, arc_centre[1] + np.sin(avg_angle) * angle_dist, r"$\beta$", horizontalalignment="center")

    try:
        print(f"B/A: {b_length / a_length:.3f}")
        print(f"alpha/beta: {alpha / beta:.3f}")
    except:
        pass