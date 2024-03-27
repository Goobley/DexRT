#if !defined(DEXRT_VOIGT_HPP)
#define DEXRT_VOIGT_HPP

#include "Types.hpp"

#if defined(YAKL_ARCH_CUDA)
#include <cuda/std/complex>
template <typename T>
using DexComplex = cuda::std::complex<T>;
#elif defined(YAKL_ARCH_HIP)
#include <thrust/complex.h>
template <typename T>
using DexComplex = thrust::complex<T>;
#else
#include <complex>
template <typename T>
using DexComplex = std::complex<T>;
#endif

namespace DexVoigtDetail {
template <typename T>
YAKL_INLINE T cexp(const T& x) {
#if defined(YAKL_ARCH_CUDA)
    return cuda::std::exp(x);
#elif defined(YAKL_ARCH_HIP)
    return thrust::exp(x);
#else
    return std::exp(x);
#endif
}
}

template <typename T=fp_t>
YAKL_INLINE DexComplex<T> humlicek_voigt(T a, T v) {
    using DexVoigtDetail::cexp;
    DexComplex<T> z(a, -v);
    T s = std::abs(v) + a;

    // NOTE(cmo): From Han's Fortran implementation of the Humlicek 1982 paper

    if (s >= FP(15.0)) {
        // Region I
        return (z * FP(0.5641896)) / (FP(0.5) + z * z);
    } else if (s >= FP(5.5)) {
        // Region II
        auto u = z * z;
        return (z * (FP(1.410474) + u*FP(0.5641896))) / (FP(0.75) + (u*(FP(3.0) + u)));
    } else if (a >= FP(0.195) * std::abs(v) - FP(0.176)) {
        // Region III
        return (FP(16.4955) + 
            z*(FP(20.20933) + z*(FP(11.96482) + z*(FP(3.778987) + FP(0.5642236)*z)))) / 
          (FP(16.4955) + z*(FP(38.82363) + z*(FP(39.27121) + z*(FP(21.69274) + 
            z*(FP(6.699398) + z)))));
    } else {
        // Region IV
        auto u = z * z;
        return cexp(u) - (z*(FP(36183.31) - u*(FP(3321.99) - u*(FP(1540.787) - 
          u*(FP(219.031) - u*(FP(35.7668) - u*(FP(1.320522) - u*FP(0.56419))))))) / 
          (FP(32066.6) - u*(FP(24322.84) - u*(FP(9022.228) - u*(FP(2186.181) - 
          u*(FP(364.2191) - u*(FP(61.57037) - u*(FP(1.841439) - u))))))));
    }
}

#else
#endif