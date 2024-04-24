import numpy as np
import matplotlib.pyplot as plt

def box_from_aabb(aabb):
    pts = np.array([
        [aabb[0, 0], aabb[1, 0]],
        [aabb[0, 1], aabb[1, 0]],
        [aabb[0, 1], aabb[1, 1]],
        [aabb[0, 0], aabb[1, 1]],
        [aabb[0, 0], aabb[1, 0]]
    ])
    return pts

box = np.array([[0, 0], [512, 0], [512, 768], [0, 768], [0, 0]])
line_seg = np.array([[0, 768], [512, 768]])
if __name__ == "__main__":
    plt.ion()
    plt.figure(figsize=(10, 10))
    plt.xlim(-256, 1024)
    plt.ylim(-256, 1024)

    plt.plot(
        box[:, 0],
        box[:, 1]
    )

    plt.plot(
        line_seg[:, 0],
        line_seg[:, 1]
    )

    rotation_point = np.mean(box[[0, 2]], axis=0)
    mu = 0.0
    mux = -np.sqrt(1.0 - mu**2)
    reverse_rotation = np.arccos(mu) * np.sign(mux)
    reverse_rot_mat = np.array([
        [np.cos(reverse_rotation), -np.sin(reverse_rotation)],
        [np.sin(reverse_rotation),  np.cos(reverse_rotation)]
    ])
    rotation = -reverse_rotation
    rot_mat = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])

    # rotated_line = np.zeros_like(line_seg)
    # for i in range(line_seg.shape[0]):
    #     rotated_line[i] = rot_mat @ (line_seg[i] - rotation_point) + rotation_point

    # plt.plot(
    #     rotated_line[:, 0],
    #     rotated_line[:, 1]
    # )

    rotated_box = np.zeros_like(box)
    for i in range(rotated_box.shape[0]):
        rotated_box[i] = reverse_rot_mat @ (box[i] - rotation_point) + rotation_point
    plt.plot(
        rotated_box[:, 0],
        rotated_box[:, 1]
    )

    rotated_box_aabb = np.array([
        [
            np.min(rotated_box[:, 0]),
            np.max(rotated_box[:, 0]),
        ],
        [
            np.min(rotated_box[:, 1]),
            np.max(rotated_box[:, 1]),
        ]
    ])
    aabb_pts = box_from_aabb(rotated_box_aabb)
    plt.plot(
        aabb_pts[:, 0],
        aabb_pts[:, 1]
    )
    aabb_max_line = np.array([
        [rotated_box_aabb[0, 0] - 1.0, rotated_box_aabb[1, 1] + 1.0],
        [rotated_box_aabb[0, 1] + 1.0, rotated_box_aabb[1, 1] + 1.0],
    ])
    rotated_aabb_line = np.zeros_like(aabb_max_line)
    for i in range(aabb_max_line.shape[0]):
        rotated_aabb_line[i] = rot_mat @ (aabb_max_line[i] - rotation_point) + rotation_point

    rotated_aabb_line[0, :] = np.floor(rotated_aabb_line[0, :])
    rotated_aabb_line[1, :] = np.ceil(rotated_aabb_line[1, :])
    plt.plot(
        rotated_aabb_line[:, 0],
        rotated_aabb_line[:, 1],
    )

    length = np.sqrt(np.sum((rotated_aabb_line[1] - rotated_aabb_line[0])**2))
    direction = (rotated_aabb_line[1] - rotated_aabb_line[0]) / length
    step = length / 4

    for i in range(4):
        start_pt = rotated_aabb_line[0] + (0.5 + i) * step * direction
        print(start_pt / rotated_aabb_line[0])
        d = np.array([-np.cos(np.deg2rad(90) + rotation), -np.sin(np.deg2rad(90) + rotation)])
        end_pt = start_pt + d * 1024
        plt.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], '--', c='C5')

    ray_dir = np.array([-mux, -mu])
    start_pt = np.array([384, 768])
    end_pt = start_pt + ray_dir * 512

    plt.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], '-.', c='r')
    # NOTE(cmo): So this should work for a mu of 30 degrees

