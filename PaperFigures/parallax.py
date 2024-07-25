import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Arc
import matplotlib as mpl
import numpy as np
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap("Set2").colors)

ADD_POST_MERGE_ROWS = True
DISCRETE_LENGTH = True
BILINEAR_MERGE = True

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    ax.set_aspect('equal')
    C0 = [(0.25, 0.25)]
    C1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    C1_labels = ['A', 'B', 'C', 'D']
    probe_radius = 0.15
    light_source_centre = (-0.6, 0.6)
    # light_source_centre = (0.5, 1.0)
    light_source_radius = 0.2
    light_source_colour = 'C5'
    C0_radius = 1.2

    def source_distance(pt_from):
        separation = np.array(light_source_centre) - np.array(pt_from)
        dist = np.sqrt(np.sum(separation**2))
        return dist

    def compute_source_wedge_angles(pt_from):
        separation = np.array(light_source_centre) - np.array(pt_from)
        dist = np.sqrt(np.sum(separation**2))
        alpha = np.arctan(light_source_radius / dist)
        central_angle = np.arctan2(separation[1], separation[0])
        s1 = np.rad2deg(central_angle - alpha)
        s2 = np.rad2deg(central_angle + alpha)
        return s1, s2


    # ax.set_xlim(-1, 1.2)
    # ax.set_ylim(-0.5, 1.7)

    source_angles_c1 = {}
    source_angles_merge = {}
    source = Circle(light_source_centre, light_source_radius, facecolor=light_source_colour)
    ax.add_patch(source)
    ax.set_axis_off()

    for c in C0:
        # probe centre
        ax.plot(c[0], c[1], 'o', c='C0')

        s1_c0, s2_c0 = compute_source_wedge_angles(c)
        if DISCRETE_LENGTH and BILINEAR_MERGE:
            for label, cc in zip(C1_labels, C1):
                if source_distance(cc) > C0_radius:
                    continue
                # c1 visibility
                s1, s2 = compute_source_wedge_angles(cc)
                source_angles_merge[label] = (s1, s2)
                source_wedge = Wedge(c, probe_radius, s1, s2, facecolor=light_source_colour, alpha=0.5)
                ax.add_patch(source_wedge)
        elif not BILINEAR_MERGE:
            # c0 visibility
            source_wedge = Wedge(c, probe_radius, s1_c0, s2_c0, facecolor=light_source_colour)
            ax.add_patch(source_wedge)

        # probe circle
        probe = Circle(c, probe_radius, fill=False, edgecolor='C0')
        ax.add_patch(probe)

        if DISCRETE_LENGTH:
            # probe end radius
            probe = Arc(c, 2*C0_radius, 2*C0_radius, theta1=-40, theta2=220, fill=False, edgecolor='C0')
            ax.add_patch(probe)

    for c, label in zip(C1, C1_labels):
        # probe centre
        ax.plot(c[0], c[1], 'x', c='C1')
        plt.text(c[0] + 0.04, c[1] - 0.05, label)

        if not DISCRETE_LENGTH or (DISCRETE_LENGTH and source_distance(c) > C0_radius):
            # c1 visibility
            s1, s2 = compute_source_wedge_angles(c)
            source_angles_c1[label] = (s1, s2)
            source_angles_merge[label] = (s1, s2)
            source_wedge = Wedge(c, probe_radius, s1, s2, facecolor=light_source_colour)
            ax.add_patch(source_wedge)

        if DISCRETE_LENGTH and BILINEAR_MERGE:
            if source_distance(c) < C0_radius:
                s1, s2 = compute_source_wedge_angles(c)
                source_wedge_block = Wedge(c, probe_radius, s1, s2, facecolor='C0', hatch='////', alpha=0.3)
                ax.add_patch(source_wedge_block)
        else:
            # c0 block
            source_wedge_block = Wedge(c, probe_radius, s1_c0, s2_c0, facecolor='C0', hatch='////', alpha=0.3)
            ax.add_patch(source_wedge_block)

        # probe circle
        probe = Circle(c, probe_radius, fill=False, edgecolor='C1')
        ax.add_patch(probe)

        if DISCRETE_LENGTH:
            # probe start radius
            probe = Arc(c, 2*C0_radius, 2*C0_radius, theta1=-40, theta2=220, fill=False, edgecolor='C1')
            ax.add_patch(probe)


    probe_type = "Finite Length" if DISCRETE_LENGTH else "Continuous"
    merge_type = "Bilinear Fix Merge" if BILINEAR_MERGE else "Standard Merge"
    title = f"{probe_type} Probes ({merge_type})"
    ax.set_title(title)
    probe_type_save = "Finite" if DISCRETE_LENGTH else "Continuous"
    merge_type_save = "Bilinear" if BILINEAR_MERGE else "Standard"
    save_name = f"Parallax_Fig_{probe_type_save}_{merge_type_save}"

    if ADD_POST_MERGE_ROWS:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

        row0_y = ylims[0] - probe_radius * 1.5
        ax.axhline(ylims[0], c="C7")
        num_merge = 5
        x_spacing = (xlims[1] - xlims[0]) / num_merge
        labels_and_merge = C1_labels + ["M"]
        weights = [0.5625, 0.1875, 0.1875, 0.0625]

        for m, label in zip(range(num_merge), labels_and_merge):
            # probe centre
            probe_centre_x = (m + 0.5) * x_spacing + xlims[0]
            probe_centre = (probe_centre_x, row0_y)
            ax.plot(probe_centre[0], probe_centre[1], 'o', c='C0')
            plt.text(probe_centre[0] + 0.04, probe_centre[1] - 0.05, label)


            if label in source_angles_c1:
                # c1 visibility
                s1, s2 = source_angles_c1[label]
                source_wedge = Wedge(probe_centre, probe_radius, s1, s2, facecolor='C1')
                ax.add_patch(source_wedge)
            elif label == "M":
                for weight, label in zip(weights, source_angles_merge):
                    weight = np.sqrt(weight)
                    s1, s2 = source_angles_merge[label]
                    source_wedge = Wedge(probe_centre, probe_radius, s1, s2, facecolor='C1', alpha=weight)
                    ax.add_patch(source_wedge)
                # # c0 visibility
                # source_wedge = Wedge(probe_centre, probe_radius, s1_c0, s2_c0, facecolor='C0')
                # ax.add_patch(source_wedge)

            if not BILINEAR_MERGE:
                # c0 visibility
                source_wedge = Wedge(probe_centre, probe_radius, s1_c0, s2_c0, facecolor='C0')
                ax.add_patch(source_wedge)

            probe = Circle(probe_centre, probe_radius, fill=False, edgecolor='C0')
            ax.add_patch(probe)


        # row1_y = row0_y - probe_radius * 2.5
        # for m, label in zip(range(num_merge), labels_and_merge):
        #     # probe centre
        #     probe_centre_x = (m + 0.5) * x_spacing + xlims[0]
        #     probe_centre = (probe_centre_x, row1_y)
        #     ax.plot(probe_centre[0], probe_centre[1], 'o', c='C0')
        #     plt.text(probe_centre[0] + 0.02, probe_centre[1] - 0.03, label)


        #     if label in source_angles_c1:
        #         # c1 visibility
        #         s1, s2 = source_angles_c1[label]
        #         source_wedge = Wedge(probe_centre, probe_radius, s1, s2, facecolor='C1')
        #         ax.add_patch(source_wedge)
        #     else:
        #         for weight, label in zip(weights, source_angles_c1):
        #             weight = np.sqrt(weight)
        #             s1, s2 = source_angles_c1[label]
        #             source_wedge = Wedge(probe_centre, probe_radius, s1, s2, facecolor='C1', alpha=weight)
        #             ax.add_patch(source_wedge)


        #     probe = Circle(probe_centre, probe_radius, fill=False, edgecolor='C0')
        #     ax.add_patch(probe)

    fig.savefig(f"{save_name}.png", dpi=300)
    fig.savefig(f"{save_name}.pdf")

