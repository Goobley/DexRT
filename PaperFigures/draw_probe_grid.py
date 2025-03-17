import numpy as np
import matplotlib.pyplot as plt
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

CANVAS_X = 512
CANVAS_Y = 512
PROBE0_LENGTH = 8
PROBE0_NUM_RAYS = 4
PROBE0_SPACING = 32
CASCADE_BRANCHING_FACTOR = 1
MAX_LEVEL = 3


if __name__ == "__main__":
    plt.ion()
    plt.figure(figsize=(4,4), constrained_layout=True)
    plt.xlim(0, CANVAS_X)
    plt.ylim(0, CANVAS_Y)

    prev_radius = 0
    for l in range(MAX_LEVEL + 1):
        radius = PROBE0_LENGTH * 2**(l *  CASCADE_BRANCHING_FACTOR)
        num_rays = PROBE0_NUM_RAYS * 2**(l * CASCADE_BRANCHING_FACTOR)
        # spacing = PROBE0_SPACING * 2**l
        spacing = PROBE0_SPACING * np.sqrt(3)**l

        x = 0.0
        while x < CANVAS_X:
            y = 0.0
            while y < CANVAS_Y:
                centre_x = x + 0.5 * spacing
                centre_y = y + 0.5 * spacing
                for ray_idx in range(num_rays):
                    angle = 2 * np.pi / num_rays * (ray_idx + 0.5)
                    plt.plot(
                        [
                            centre_x + prev_radius * np.cos(angle),
                            centre_x + radius * np.cos(angle),
                        ],
                        [
                            centre_y + prev_radius * np.sin(angle),
                            centre_y + radius * np.sin(angle),
                        ],
                        c=f'C{l}',
                        lw=0.7,
                    )
                y += spacing
            x += spacing

        prev_radius = radius

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_frame_on(False)
    # plt.savefig("ProbeGrid.pdf")
    # plt.savefig("ProbeGrid.png", dpi=300)