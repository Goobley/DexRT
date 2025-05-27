import numpy as np
mux_A2 = np.array([0.57735026])
muy_A2 = np.array([0.57735026])
wmu_A2 = np.array([1.00000000])

mux_A4 = np.array([0.88191710, 0.33333333, 0.33333333])
muy_A4 = np.array([0.33333333, 0.88191710, 0.33333333])
wmu_A4 = np.array([0.33333333, 0.33333333, 0.33333333])

mux_A6 = np.array([0.93094934, 0.68313005, 0.25819889, 0.68313005, 0.25819889, 0.25819889])
muy_A6 = np.array([0.25819889, 0.68313005, 0.93094934, 0.25819889, 0.68313005, 0.25819889])
wmu_A6 = np.array([0.18333333, 0.15000000, 0.18333333, 0.15000000, 0.15000000, 0.18333333])

mux_A8 = np.array([0.95118973, 0.78679579, 0.57735027, 0.21821789, 0.78679579, 0.57735027, 0.21821789, 0.57735027, 0.21821789, 0.21821789])
muy_A8 = np.array([0.21821789, 0.57735027, 0.78679579, 0.95118973, 0.21821789, 0.57735027, 0.78679579, 0.21821789, 0.57735027, 0.21821789])
wmu_A8 = np.array([0.12698138, 0.09138353, 0.09138353, 0.12698138, 0.09138353, 0.07075469, 0.09138353, 0.09138353, 0.09138353, 0.12698138])

QUAD_SET = "A8"

if __name__ == "__main__":
    mux_in = locals()[f"mux_{QUAD_SET}"]
    muy_in = locals()[f"muy_{QUAD_SET}"]
    wmu_in = locals()[f"wmu_{QUAD_SET}"]

    Nmu = mux_in.shape[0] * 8
    mux = np.zeros(Nmu)
    muy = np.zeros(Nmu)
    muz = np.zeros(Nmu)
    wmu = np.zeros(Nmu)

    mux[:int(Nmu / 8)] = mux_in
    muy[:int(Nmu / 8)] = muy_in
    for m in range(Nmu):
        wmu[m] = wmu_in[m % int(Nmu / 8)]
    wmu /= np.sum(wmu)

    # y-z reflection
    mux[int(Nmu / 8) : int(Nmu / 4)] = -mux[: int(Nmu / 8)]
    muy[int(Nmu / 8) : int(Nmu / 4)] =  muy[: int(Nmu / 8)]

    # x-z reflection
    mux[int(Nmu / 4) : int(Nmu / 2)] =  mux[: int(Nmu / 4)]
    muy[int(Nmu / 4) : int(Nmu / 2)] = -muy[: int(Nmu / 4)]

    # compute muz
    muz = np.sqrt(1.0 - mux**2 - muy**2)

    # x-y reflection
    mux[int(Nmu / 2):] =  mux[:int(Nmu / 2)]
    muy[int(Nmu / 2):] =  muy[:int(Nmu / 2)]
    muz[int(Nmu / 2):] = -muz[:int(Nmu / 2)]

    # output
    print(f"// Set {QUAD_SET}")
    print(f"constexpr i32 NUM_LC_QUAD = {Nmu};")
    print(f"constexpr fp_t LC_QUAD_X[NUM_LC_QUAD] = {{ {", ".join(["FP(%.7e)" % m for m in mux])} }};")
    print(f"constexpr fp_t LC_QUAD_Y[NUM_LC_QUAD] = {{ {", ".join(["FP(%.7e)" % m for m in muy])} }};")
    print(f"constexpr fp_t LC_QUAD_Z[NUM_LC_QUAD] = {{ {", ".join(["FP(%.7e)" % m for m in muz])} }};")
    print(f"constexpr fp_t LC_WEIGHT[NUM_LC_QUAD] = {{ {", ".join(["FP(%.7e)" % m for m in wmu])} }};")
