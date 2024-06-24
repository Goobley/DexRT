#if !defined(DEXRT_CONSTANTS_HPP)
#define DEXRT_CONSTANTS_HPP

#include "Config.hpp"

// NOTE(cmo): All constants from astropy, i.e. CODATA2018
namespace ConstantsF64 {
    constexpr f64 c = 299792458.0; // [m / s]
    constexpr f64 h = 6.62607015e-34; // [J s]
    constexpr f64 hc = 1.986445857148928629e-25; // [J m]
    constexpr f64 hc_kJ_nm = 1.986445857148928629e-19; // [kJ nm]
    constexpr f64 twohc2_kW_nm2 = 0.11910429723971883; // [kW nm2]
    constexpr f64 eV = 1.602176633999999894e-19; // [J]
    constexpr f64 hc_eV = 1.239841984332002624e-06; // [eV m]
    constexpr f64 hc_eV_nm = 1.239841984332002539e+03; // [eV nm]
    constexpr f64 hc_k_B_nm = 14387768.775039336; // [K nm]
    constexpr f64 u = 1.660539066599999971e-27; // [kg]
    constexpr f64 m_e = 9.1093837015e-31; // [kg]
    constexpr f64 k_B = 1.380649e-23; // [J / K]
    constexpr f64 k_B_eV =  8.61733326e-05; // [eV / K]
    constexpr f64 k_B_u = 8314.46262102654; // [J / (K kg)]
    constexpr f64 pi = 3.14159265358979312;
    constexpr f64 four_pi = 4.0 * pi;
    constexpr f64 sqrt_pi = 1.7724538509055159;
    constexpr f64 seaton_c0 = 8.629132180819956e-12; // [m2 J(1/2) K(1/2) / kg(1/2)]
}

namespace ConstantsFP {
    constexpr fp_t c = FP(299792458.0); // [m / s]
    constexpr fp_t h = FP(6.62607015e-34); // [J s]
    constexpr fp_t hc = FP(1.986445857148928629e-25); // [J m]
    constexpr fp_t hc_kJ_nm = FP(1.986445857148928629e-19); // [kJ nm]
    constexpr fp_t twohc2_kW_nm2 = FP(0.11910429723971883); // [kW nm2]
    constexpr fp_t eV = FP(1.602176633999999894e-19); // [J]
    constexpr fp_t hc_eV = FP(1.239841984332002624e-06); // [eV m]
    constexpr fp_t hc_eV_nm = FP(1.239841984332002539e+03); // [eV nm]
    constexpr fp_t hc_k_B_nm = FP(14387768.775039336); // [K nm]
    constexpr fp_t u = FP(1.660539066599999971e-27); // [kg]
    constexpr fp_t m_e = FP(9.1093837015e-31); // [kg]
    constexpr fp_t k_B = FP(1.380649e-23); // [J / K]
    constexpr fp_t k_B_eV =  FP(8.61733326e-05); // [eV / K]
    constexpr fp_t k_B_u = FP(8314.46262102654); // [J / (K kg)]
    constexpr f64 pi = FP(3.14159265358979312);
    constexpr fp_t four_pi = FP(4.0) * pi;
    constexpr fp_t sqrt_pi = FP(1.7724538509055159);
    constexpr fp_t seaton_c0 = FP(8.629132180819956e-12); // [m2 J(1/2) K(1/2) / kg(1/2)]
}

#else
#endif
