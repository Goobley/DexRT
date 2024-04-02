import numpy as np
import astropy.units as u
import astropy.constants as const

NUM_BOUND_LEVELS = 5
TEMPERATURES = np.logspace(3, 7, 31) << u.K

if __name__ == "__main__":
    bound_n = np.arange(NUM_BOUND_LEVELS) + 1
    g_n = 2 * bound_n**2
    En = const.Ryd.to(u.J, equivalencies=u.spectral()) * (1.0 - 1.0 / bound_n**2)
    U_I_grid = g_n[None, :] * np.exp(-(En / const.k_B)[None, :] / TEMPERATURES[:, None])
    U_I_temperature = np.sum(U_I_grid, axis=1)

    log_T = np.log10(TEMPERATURES.value)
    log_T_repr = "constexpr fp_t log_T[] = {"
    for i, t in enumerate(log_T):
        log_T_repr += f"FP({t:e})"
        if i != log_T.shape[0] - 1:
            log_T_repr += ", "
    log_T_repr += "};"

    h_part_fn = U_I_temperature
    h_part_fn_repr = "constexpr fp_t h_partfn[] = {"
    for i, t in enumerate(h_part_fn):
        h_part_fn_repr += f"FP({t:e})"
        if i != h_part_fn.shape[0] - 1:
            h_part_fn_repr += ", "
    h_part_fn_repr += "};"

    print(log_T_repr)
    print(h_part_fn_repr)