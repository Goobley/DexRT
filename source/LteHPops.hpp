#if !defined(DEXRT_LTE_H_POPS_HPP)
#define DEXRT_LTE_H_POPS_HPP

#include "Types.hpp"
#include "Utils.hpp"

/**
 * Temporary function to give us the LTE ground state fraction of h for
 * broadening things. Uses tabulated partition function based on classical 5+1
 * level H.
 *
 * N.B. This seems to be able to overflow the stack in deep functions - use a properly allocated one!!
*/
YAKL_INLINE fp_t nh0_lte(fp_t temperature, fp_t ne, fp_t nh_tot, fp_t* nhii=nullptr) {
    constexpr int grid_size = 31;
    constexpr fp_t log_T[grid_size] = {FP(3.000000e+00), FP(3.133333e+00), FP(3.266667e+00), FP(3.400000e+00), FP(3.533333e+00), FP(3.666667e+00), FP(3.800000e+00), FP(3.933333e+00), FP(4.066667e+00), FP(4.200000e+00), FP(4.333333e+00), FP(4.466667e+00), FP(4.600000e+00), FP(4.733333e+00), FP(4.866667e+00), FP(5.000000e+00), FP(5.133333e+00), FP(5.266667e+00), FP(5.400000e+00), FP(5.533333e+00), FP(5.666667e+00), FP(5.800000e+00), FP(5.933333e+00), FP(6.066667e+00), FP(6.200000e+00), FP(6.333333e+00), FP(6.466667e+00), FP(6.600000e+00), FP(6.733333e+00), FP(6.866667e+00), FP(7.000000e+00)};
    constexpr fp_t h_partfn[grid_size] = {FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000012e+00), FP(2.000628e+00), FP(2.013445e+00), FP(2.136720e+00), FP(2.776522e+00), FP(4.825934e+00), FP(9.357075e+00), FP(1.691968e+01), FP(2.713700e+01), FP(3.892452e+01), FP(5.101649e+01), FP(6.238607e+01), FP(7.240922e+01), FP(8.083457e+01), FP(8.767237e+01), FP(9.307985e+01), FP(9.727538e+01), FP(1.004851e+02), FP(1.029155e+02), FP(1.047417e+02), FP(1.061062e+02), FP(1.071217e+02), FP(1.078750e+02), FP(1.084327e+02)};

    using namespace ConstantsFP;
    constexpr fp_t saha_const = (FP(2.0) * pi * k_B) / h * (h / m_e);
    constexpr fp_t ion_energy_k_B = FP(157887.51240204);
    constexpr fp_t u_hii = FP(1.0);

    const KView<const fp_t*> log_t_grid((fp_t*)log_T, grid_size);
    const KView<const fp_t*> partfn_grid((fp_t*)h_partfn, grid_size);
    const fp_t log_temp = std::log10(temperature);
    const fp_t u_hi = interp(log_temp, log_t_grid, partfn_grid);
    // NOTE(cmo): nhii / nhi
    const fp_t ratio = 2 * u_hii / (ne * u_hi) * std::pow(saha_const * temperature, FP(1.5)) * std::exp(-ion_energy_k_B / temperature);
    const fp_t result = nh_tot / (FP(1.0) + ratio);
    if (nhii) {
        *nhii = ratio * result;
    }
    return result;
}


namespace LteHPopsDetail {
    constexpr int grid_size = 31;
    constexpr fp_t log_T_data[grid_size] = {FP(3.000000e+00), FP(3.133333e+00), FP(3.266667e+00), FP(3.400000e+00), FP(3.533333e+00), FP(3.666667e+00), FP(3.800000e+00), FP(3.933333e+00), FP(4.066667e+00), FP(4.200000e+00), FP(4.333333e+00), FP(4.466667e+00), FP(4.600000e+00), FP(4.733333e+00), FP(4.866667e+00), FP(5.000000e+00), FP(5.133333e+00), FP(5.266667e+00), FP(5.400000e+00), FP(5.533333e+00), FP(5.666667e+00), FP(5.800000e+00), FP(5.933333e+00), FP(6.066667e+00), FP(6.200000e+00), FP(6.333333e+00), FP(6.466667e+00), FP(6.600000e+00), FP(6.733333e+00), FP(6.866667e+00), FP(7.000000e+00)};
    constexpr fp_t h_partfn_data[grid_size] = {FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000012e+00), FP(2.000628e+00), FP(2.013445e+00), FP(2.136720e+00), FP(2.776522e+00), FP(4.825934e+00), FP(9.357075e+00), FP(1.691968e+01), FP(2.713700e+01), FP(3.892452e+01), FP(5.101649e+01), FP(6.238607e+01), FP(7.240922e+01), FP(8.083457e+01), FP(8.767237e+01), FP(9.307985e+01), FP(9.727538e+01), FP(1.004851e+02), FP(1.029155e+02), FP(1.047417e+02), FP(1.061062e+02), FP(1.071217e+02), FP(1.078750e+02), FP(1.084327e+02)};
}
/**
 * Temporary object to give us the LTE ground state fraction of h for
 * broadening things. Uses tabulated partition function based on classical 5+1
 * level H.
*/
template <typename mem_space=DefaultMemSpace>
struct HPartFn {
    KView<const fp_t*, mem_space> log_T;
    KView<const fp_t*, mem_space> h_partfn;

    HPartFn() {
        using namespace LteHPopsDetail;
        KView<fp_t*, HostSpace> log_T_host((fp_t*)log_T_data, grid_size);
        KView<fp_t*, HostSpace> partfn_host((fp_t*)h_partfn_data, grid_size);
        if constexpr (std::is_same_v<mem_space, HostSpace>) {
            log_T = log_T_host;
            h_partfn = partfn_host;
        } else {
            log_T = create_device_copy(log_T_host);
            h_partfn = create_device_copy(partfn_host);
        }
    }

    YAKL_INLINE fp_t operator()(fp_t temperature, fp_t ne, fp_t nh_tot, fp_t* nhii=nullptr) const {
#if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
        KOKKOS_IF_ON_HOST(
            if constexpr (!Kokkos::SpaceAccessibility<mem_space, DefaultMemSpace>::accessible) {
                throw std::runtime_error(fmt::format("Cannot access a partition function in device memory from CPU..."));
            }
        );
#endif
        using namespace ConstantsFP;
        constexpr fp_t saha_const = (FP(2.0) * pi * k_B) / h * (h / m_e);
        constexpr fp_t ion_energy_k_B = FP(157887.51240204);
        constexpr fp_t u_hii = FP(1.0);
        const fp_t log_temp = std::log10(temperature);
        const fp_t u_hi = interp(log_temp, log_T, h_partfn);
        // NOTE(cmo): nhii / nhi
        const fp_t ratio = 2 * u_hii / (ne * u_hi) * std::pow(saha_const * temperature, FP(1.5)) * std::exp(-ion_energy_k_B / temperature);
        const fp_t result = nh_tot / (FP(1.0) + ratio);
        if (nhii) {
            *nhii = ratio * result;
        }
        return result;
    }
};
#else
#endif