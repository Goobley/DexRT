import numpy as np
# https://mathworld.wolfram.com/LobattoQuadrature.html
GL_4 = np.array([-1.0, -1.0/5.0*np.sqrt(5), 1.0/5.0*np.sqrt(5), 1.0])
WGL_4 = np.array([1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0])

GL = GL_4
WGL = WGL_4

mid, half_width = 0.5, 0.5
mu = mid + half_width * GL_4
w = WGL_4 * half_width

mus = ", ".join(["FP(%f)" % m for m in reversed(mu)])
incl_mus = ", ".join(["FP(%f)" % (np.sqrt(1 - m**2)) for m in reversed(mu)])
ws = ", ".join(["FP(%f)" % weight for weight in reversed(w)])
# NOTE(cmo): Chop off zero mu weight -- special handling in code.
reduced_mus = ", ".join(["FP(%f)" % m for m in reversed(mu[1:])])
reduced_incl_mus = ", ".join(["FP(%f)" % (np.sqrt(1 - m**2)) for m in reversed(mu[1:])])
reduced_ws = ", ".join(["FP(%f)" % weight for weight in reversed(w[1:])])
print(f"constexpr int NUM_AZ = {GL_4.shape[0]};")
print("constexpr int NUM_GAUSS_LOBATTO = yakl::max(NUM_AZ - 1, 1);")
# print(f"constexpr yakl::SArray<fp_t, 1, NUM_GAUSS_LOBATTO> AZ_RAYS = {{{mus}}};")
# print(f"constexpr yakl::SArray<fp_t, 1, NUM_GAUSS_LOBATTO> AZ_WEIGHTS = {{{ws}}};")
print(f"constexpr fp_t TRACE_AZ_RAYS[NUM_GAUSS_LOBATTO] = {{{reduced_mus}}};")
print(f"constexpr fp_t TRACE_INCL_RAYS[NUM_GAUSS_LOBATTO] = {{{reduced_incl_mus}}};")
print(f"constexpr fp_t TRACE_AZ_WEIGHTS[NUM_GAUSS_LOBATTO] = {{{reduced_ws}}};")
print(f"constexpr fp_t AZ_RAYS[NUM_AZ] = {{{mus}}};")
print(f"constexpr fp_t INCL_RAYS[NUM_AZ] = {{{incl_mus}}};")
print(f"constexpr fp_t AZ_WEIGHTS[NUM_AZ] = {{{ws}}};")
