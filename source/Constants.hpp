#if !defined(DEXRT_CONSTANTS_HPP)
#define DEXRT_CONSTATNTS_HPP

#include "Config.hpp"

// NOTE(cmo): All constants from astropy, i.e. CODATA2018
namespace ConstantsF64 {
    constexpr f64 c = 299792458.0; // [m / s]
    constexpr f64 h = 6.62607015e-34; // [J s]
    constexpr f64 hc = 1.986445857148928629e-25; // [J m]
    constexpr f64 eV = 1.602176633999999894e-19; // [J]
    constexpr f64 hc_eV = 1.239841984332002624e-06; // [eV m]
    constexpr f64 hc_eV_nm = 1.239841984332002539e+03; // [eV nm]
    constexpr f64 u = 1.660539066599999971e-27; // [kg]
}

namespace ConstantsFP {
    constexpr fp_t c = FP(299792458.0); // [m / s]
    constexpr fp_t h = FP(6.62607015e-34); // [J s]
    constexpr fp_t hc = FP(1.986445857148928629e-25); // [J m]
    constexpr fp_t eV = FP(1.602176633999999894e-19); // [J]
    constexpr fp_t hc_eV = FP(1.239841984332002624e-06); // [eV m]
    constexpr fp_t hc_eV_nm = FP(1.239841984332002539e+03); // [eV nm]
    constexpr fp_t u = FP(1.660539066599999971e-27); // [kg]
}

#else
#endif