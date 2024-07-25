import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Arc
import matplotlib as mpl
import numpy as np
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap("Set2").colors)

CENTRED = False

def gamma_centred(w, d):
    v = -w/2
    sin_gamma = w * d / np.sqrt((d**2 + v**2)**2 + (d**2 + v**2)*w*(2*v+w))
    return np.arcsin(sin_gamma)

def gamma_offset(w, d, v):
    sin_gamma = (w * d) / np.sqrt((d**2 + v**2) * (d**2 + (v+w)**2))
    return np.arcsin(sin_gamma)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.set_aspect("equal")
    ax.set_axis_off()
    CANVAS_SIZE = (4.0, 4.0)
    ax.set_xlim(0.0, CANVAS_SIZE[0])
    ax.set_ylim(0.0, CANVAS_SIZE[1]+0.01)
    source_size = 2.0
    blocker_perp_dist = 1.0
    blocker_y = CANVAS_SIZE[1] - blocker_perp_dist
    source_offset = 0.0
    if CENTRED:
        source_offset = -0.5 * source_size

    # draw source
    source_start = 0.5 * CANVAS_SIZE[0] + source_offset
    source_end = source_start + source_size
    plt.plot([source_start, source_end], [CANVAS_SIZE[1], CANVAS_SIZE[1]], lw=3, c="C5")
    # draw blocker
    blocker_start = 0.0
    blocker_end = 0.5 * CANVAS_SIZE[0]
    plt.plot([blocker_start, blocker_end], [blocker_y, blocker_y], lw=4, c='k')

    line1 = np.array([blocker_end, blocker_y]) - np.array([source_start, CANVAS_SIZE[1]])
    line1_dir = line1 / np.linalg.norm(line1)
    line2 = np.array([blocker_end, blocker_y]) - np.array([source_end, CANVAS_SIZE[1]])
    line2_dir = line2 / np.linalg.norm(line2)
    line_angle = np.arccos((line1 @ line2) / (np.linalg.norm(line1) * np.linalg.norm(line2)))
    pred_angle = gamma_offset(source_size, blocker_perp_dist, source_offset)

    t1_max = (0.0 - CANVAS_SIZE[1]) / line1_dir[1]
    t2_max = (0.0 - CANVAS_SIZE[1]) / line2_dir[1]

    plt.plot([source_start, source_start + t1_max * line1_dir[0]], [CANVAS_SIZE[1], CANVAS_SIZE[1] + t1_max * line1_dir[1]], '--')
    plt.plot([source_end, source_end + t2_max * line2_dir[0]], [CANVAS_SIZE[1], CANVAS_SIZE[1] + t2_max * line2_dir[1]], '--')