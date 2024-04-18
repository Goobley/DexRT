import numpy as np
import promweaver as pw
import astropy.constants as const

def float32(x):
    return np.float32(x)
dtype = np.float32

Rs = float32(const.R_sun.value)
height = float32(1.0e6)

p0 = np.array([0.0, 0.0, height + Rs], dtype=dtype)
p1 = np.array([5.0e6, 0.0, height + Rs], dtype=dtype)

def classical_dilution(h):
    Rs = const.R_sun.value
    return 0.5 * (1.0 - np.sqrt(1.0 - Rs**2 / (Rs + h)**2))

def compute_intersection_t(p0, d):
    # NOTE(cmo): Units of Rs
    a = float32(1.0)
    b = float32(2.0) * (p0 @ d)
    c = p0 @ p0 - float32(1.0)
    delta = b*b - float32(4.0) * a * c

    if delta < 0:
        return None
    return (-b - np.sqrt(delta)) / (float32(2.0) * a)

def compute_intersection_angle(p0, d):
    # NOTE(cmo): Units of Rs
    p0 = p0.copy()
    p0 /= Rs
    t_min = compute_intersection_t(p0, d)
    if t_min is None:
        return None
    int_loc = p0 + t_min * d
    # cosphi = -(d @ int_loc) / Rs
    # NOTE(cmo): Rs is 1.0
    cosphi = -(d @ int_loc)
    return cosphi


if __name__ == "__main__":
    muz = float32(0.7)
    x_frac = float32(1.0)
    mu = np.array(
        [
            np.sqrt(x_frac * (1.0 - muz**2)),
            # np.sqrt((1.0 - x_frac) * (1.0 - muz**2)),
            0.0,
            -muz
        ],
        dtype=dtype,
    )
    pw_sol = pw.outgoing_chromo_ray_mu(muz, height)
    aligned_sol = compute_intersection_angle(p0, mu)
    offset_sol = compute_intersection_angle(p1, mu)
    mu_mirrored = mu.copy()
    mu_mirrored[0] = -mu_mirrored[0]
    aligned_mirror = compute_intersection_angle(p0, mu_mirrored)
    offset_mirror = compute_intersection_angle(p1, mu_mirrored)
    pw_1Mm = pw.outgoing_chromo_ray_mu(muz, 1.0e6)
    p0_1Mm_t = ((Rs + 1.0e6) - p0[2]) / mu[2]
    p0_prime = p0 + p0_1Mm_t * mu
    p1_1Mm_t = ((Rs + 1.0e6) - p1[2]) / mu[2]
    p1_prime = p1 + p1_1Mm_t * mu
    aligned_1Mm = compute_intersection_angle(p0_prime, mu)
    offset_1Mm = compute_intersection_angle(p1_prime, mu)
    print(f"pw: {pw_sol}, this method: {aligned_sol}")
    print(f"with a {p1[0]/1e6} Mm offset @ an altitude of {height/1e6} Mm: {offset_sol}")
    print(f"with a mirrored offset: {offset_mirror} (aligned: {aligned_mirror})")
    print(f"=============================")
    print(f"Launching from downstream plane")
    print(f"pw: {pw_1Mm}, this method: {aligned_1Mm}")
    print(f"offset: {offset_1Mm}")


    print("===================\n\n")
    Nrays = 1024
    accum = 0.0
    for r in range(Nrays):
        angle = 2.0 * np.pi * (r + 0.5) / Nrays
        direction = np.array([
            np.cos(angle),
            0.0,
            np.sin(angle),
        ])

        if direction[2] < 0.0:
            mu_chromo = compute_intersection_angle(p1, direction)
            if mu_chromo is not None:
                curr_weight = np.abs(direction[0])
                accum += curr_weight * 1.0 * (0.5 * np.pi)

    accum /= Nrays #* (2.0 / np.pi)

    classical = classical_dilution(height)
    print(f"Got dilution {accum}, expected {classical}, ratio: {accum / classical}")
    subtended = 0.5 * (1.0 - np.sqrt((Rs + height)**2 - Rs**2) / (Rs + height))
    print(f"Subtended/4pi = {subtended}")


