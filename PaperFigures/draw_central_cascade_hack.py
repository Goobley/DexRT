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

CANVAS_X = 128
CANVAS_Y = 128
PROBE0_LENGTH = 8
PROBE0_NUM_RAYS = 4
CASCADE_BRANCHING_FACTOR = 1
MAX_LEVEL = 3
BRANCH_RAYS = False
SPLIT_QUADRANTS = True

highlight_path = [0, 0, 0, 0]
highlight_colour = "C6"
highlight_ls = "--"
highlight_entries = []

def do_the_draw():
    prev_radius = 0
    for l in range(MAX_LEVEL + 1):
        radius = PROBE0_LENGTH * 2**(l * CASCADE_BRANCHING_FACTOR)

        num_rays = PROBE0_NUM_RAYS * 2**(l * CASCADE_BRANCHING_FACTOR)
        prev_num_rays = 1
        if l > 0:
            prev_num_rays = PROBE0_NUM_RAYS * 2**((l-1) * CASCADE_BRANCHING_FACTOR)

        num_rays_to_plot = num_rays
        # num_rays_to_plot = 2**(l * CASCADE_BRANCHING_FACTOR)
        for ray_idx in range(num_rays_to_plot):
            angle = 2 * np.pi / num_rays * (ray_idx + 0.5)
            if BRANCH_RAYS:
                prev_idx = ray_idx // 2**CASCADE_BRANCHING_FACTOR
                prev_angle = 2 * np.pi / prev_num_rays * (prev_idx + 0.5)
                start = [
                    centre_x + prev_radius * np.cos(prev_angle),
                    centre_y + prev_radius * np.sin(prev_angle)
                ]
            else:
                start = [
                    centre_x + prev_radius * np.cos(angle),
                    centre_y + prev_radius * np.sin(angle)
                ]

            colour = f"C{l}"
            ls = "-"
            if len(highlight_path) > l and highlight_path[l] == ray_idx:
                colour = highlight_colour
                ls = highlight_ls
                highlight_entries.append(start)
                highlight_entries.append([centre_x + radius * np.cos(angle), centre_x + radius * np.sin(angle)])
                plt.plot(
                    [
                        start[0],
                        centre_x + radius * np.cos(angle),
                    ],
                    [
                        start[1],
                        centre_y + radius * np.sin(angle),
                    ],
                    c=colour,
                    ls=ls,
                    alpha=0.9,
                    lw=1.2,
                )

        prev_radius = radius

    if len(highlight_entries) > 0:
        highlights = highlight_entries[1:-1]
        for h in range(0, len(highlights)-1, 2):
            start = highlights[h]
            end = highlights[h+1]
            plt.plot([start[0], end[0]], [start[1], end[1]], c=highlight_colour, ls=highlight_ls)



if __name__ == "__main__":
    plt.ion()
    plt.figure(figsize=(4, 4), constrained_layout=True)
    plt.xlim(0, CANVAS_X)
    plt.ylim(0, CANVAS_Y)

    centre_x = CANVAS_X // 2
    centre_y = CANVAS_Y // 2

    prev_radius = 0
    for l in range(MAX_LEVEL + 1):
        radius = PROBE0_LENGTH * 2**(l * CASCADE_BRANCHING_FACTOR)

        num_rays = PROBE0_NUM_RAYS * 2**(l * CASCADE_BRANCHING_FACTOR)
        prev_num_rays = 1
        if l > 0:
            prev_num_rays = PROBE0_NUM_RAYS * 2**((l-1) * CASCADE_BRANCHING_FACTOR)

        num_rays_to_plot = num_rays
        # num_rays_to_plot = 2**(l * CASCADE_BRANCHING_FACTOR)
        for ray_idx in range(num_rays_to_plot):
            angle = 2 * np.pi / num_rays * (ray_idx + 0.5)
            if BRANCH_RAYS:
                prev_idx = ray_idx // 2**CASCADE_BRANCHING_FACTOR
                prev_angle = 2 * np.pi / prev_num_rays * (prev_idx + 0.5)
                start = [
                    centre_x + prev_radius * np.cos(prev_angle),
                    centre_y + prev_radius * np.sin(prev_angle)
                ]
            else:
                start = [
                    centre_x + prev_radius * np.cos(angle),
                    centre_y + prev_radius * np.sin(angle)
                ]

            colour = f"C{l}"
            ls = "-"
            # if len(highlight_path) > l and highlight_path[l] == ray_idx:
            #     colour = highlight_colour
            #     ls = highlight_ls
            #     highlight_entries.append(start)
            #     highlight_entries.append([centre_x + radius * np.cos(angle), centre_x + radius * np.sin(angle)])
            plt.plot(
                [
                    start[0],
                    centre_x + radius * np.cos(angle),
                ],
                [
                    start[1],
                    centre_y + radius * np.sin(angle),
                ],
                c=colour,
                ls=ls,
                lw=0.7,
            )

        prev_radius = radius

    # if len(highlight_entries) > 0:
    #     highlights = highlight_entries[1:-1]
    #     for h in range(0, len(highlights)-1, 2):
    #         start = highlights[h]
    #         end = highlights[h+1]
    #         plt.plot([start[0], end[0]], [start[1], end[1]], c=highlight_colour, ls=highlight_ls)

    highlight_path = [2, 4, 8, 16]
    highlight_entries = []
    do_the_draw()

    highlight_path = [2, 4, 8, 17]
    highlight_entries = []
    do_the_draw()

    highlight_path = [2, 4, 9, 18]
    highlight_entries = []
    do_the_draw()

    highlight_path = [2, 4, 9, 19]
    highlight_entries = []
    do_the_draw()

    highlight_path = [2, 5, 10, 20]
    highlight_entries = []
    do_the_draw()

    highlight_path = [2, 5, 10, 21]
    highlight_entries = []
    do_the_draw()

    highlight_path = [2, 5, 11, 22]
    highlight_entries = []
    do_the_draw()

    highlight_path = [2, 5, 11, 23]
    highlight_entries = []
    do_the_draw()

    highlight_path = [0, 0, 1, 3]
    highlight_colour = "#444444"
    highlight_entries = []
    do_the_draw()


    if SPLIT_QUADRANTS:
        plt.axhline(centre_y, c="#cccccc", ls="--", lw=0.9)
        plt.axvline(centre_x, c="#cccccc", ls="--", lw=0.9)

    # plt.gca().tick_params(which="both", )
    # plt.gca().set_xticklabels([])
    # plt.gca().set_yticklabels([])
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    # plt.gca().set_frame_on(False)
    plt.gca().set_axis_off()

    plt.text(128, 128, "0", horizontalalignment="right", verticalalignment="top")
    plt.text(0, 128, "1", horizontalalignment="left", verticalalignment="top")
    plt.text(0, 0, "2", horizontalalignment="left", verticalalignment="bottom")
    plt.text(128, 0, "3", horizontalalignment="right", verticalalignment="bottom")

    plt.savefig("CentralInterpWithHighlight.png", dpi=300)
    plt.savefig("CentralInterpWithHighlight.pdf")

