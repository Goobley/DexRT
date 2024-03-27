#if !defined(DEXRT_VOIGT_HPP)
#define DEXRT_VOIGT_HPP

#include "Types.hpp"
#include <fmt/core.h>

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
#define FPT(X) T(FP(X))
    using DexVoigtDetail::cexp;
    DexComplex<T> z(a, -v);
    T s = std::abs(v) + a;

    // NOTE(cmo): From Han's Fortran implementation of the Humlicek 1982 paper

    if (s >= FPT(15.0)) {
        // Region I
        return (z * FPT(0.5641896)) / (FPT(0.5) + z * z);
    } else if (s >= FPT(5.5)) {
        // Region II
        auto u = z * z;
        return (z * (FPT(1.410474) + u*FPT(0.5641896))) / (FPT(0.75) + (u*(FPT(3.0) + u)));
    } else if (a >= FPT(0.195) * std::abs(v) - FPT(0.176)) {
        // Region III
        return (FPT(16.4955) + 
            z*(FPT(20.20933) + z*(FPT(11.96482) + z*(FPT(3.778987) + FPT(0.5642236)*z)))) / 
          (FPT(16.4955) + z*(FPT(38.82363) + z*(FPT(39.27121) + z*(FPT(21.69274) + 
            z*(FPT(6.699398) + z)))));
    } else {
        // Region IV
        auto u = z * z;
        return cexp(u) - (z*(FPT(36183.31) - u*(FPT(3321.99) - u*(FPT(1540.787) - 
          u*(FPT(219.031) - u*(FPT(35.7668) - u*(FPT(1.320522) - u*FPT(0.56419))))))) / 
          (FPT(32066.6) - u*(FPT(24322.84) - u*(FPT(9022.228) - u*(FPT(2186.181) - 
          u*(FPT(364.2191) - u*(FPT(61.57037) - u*(FPT(1.841439) - u))))))));
    }
#undef FPT
}

namespace DexVoigtDetail {
    template <typename T>
    struct Linspace {
        T min;
        T max;
        i32 n;
    };

    template <typename T=fp_t, bool Complex=true>
    struct ComplexOrReal {
        static YAKL_INLINE
        std::conditional_t<Complex, DexComplex<T>, T> value(const DexComplex<T>& x) {
            return x;
        }
    };

    template <typename T>
    struct ComplexOrReal<T, false> {
        static YAKL_INLINE
        T value(const DexComplex<T>& x) {
            return x.real();
        }
    };
}

template <typename T=fp_t, bool Complex=false, int mem_space=yakl::memDevice>
struct VoigtProfile {
    /// Voigt function interpolator
    /// Assumes a and v are linearly interpolated on linspaces, so
    /// these are implicit, rather than stored.
    /// Return value clamped to domain if sampled outside.
    using Linspace = DexVoigtDetail::Linspace<T>;
    typedef std::conditional_t<Complex, DexComplex<T>, T> Voigt_t;
    Linspace a_range;
    T a_step;
    Linspace v_range;
    T v_step;
    yakl::Array<Voigt_t const, 2, mem_space> samples;

    VoigtProfile() {};
    VoigtProfile(Linspace _a_range, Linspace _v_range) :
        a_range(_a_range),
        v_range(_v_range) 
    {
        a_step = (a_range.max - a_range.min) / T(a_range.n - 1);
        v_step = (v_range.max - v_range.min) / T(v_range.n - 1);
        // NOTE(cmo): Fill table
        compute_samples();
    }


    void compute_samples() {
        // NOTE(cmo): Allocate storage
        yakl::Array<Voigt_t, 2, mem_space> mut_samples("Voigt Samples", a_range.n, v_range.n);

        auto voigt_sample = YAKL_CLASS_LAMBDA (int ia, int iv) {
            T a = a_range.min + ia * a_step;
            T v = v_range.min + iv * v_step;
            auto sample = humlicek_voigt(a, v);
            return sample;
        };

        // NOTE(cmo): Compute storage
        // Man, C++ makes this a pain sometimes... unless I'm just being an idiot.
        using result_t = DexVoigtDetail::ComplexOrReal<T, Complex>;
        if constexpr (mem_space == yakl::memDevice) {
            parallel_for(
                "compute voigt", 
                SimpleBounds<2>(a_range.n, v_range.n),
                YAKL_LAMBDA (int ia, int iv) {
                    mut_samples(ia, iv) = result_t::value(voigt_sample(ia, iv));
                }
            );
        } else {
            for (int ia = 0; ia < a_range.n; ++ia) {
                for (int iv = 0; iv < v_range.n; ++iv) {
                    mut_samples(ia, iv) = result_t::value(voigt_sample(ia, iv));
                }
            }
        }
        samples = mut_samples;
    }

    /// Simple clamped bilinear lookup
    YAKL_INLINE Voigt_t operator()(T a, T v) const {
#if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
        YAKL_EXECUTE_ON_HOST_ONLY(
            if constexpr (mem_space == memDevice) {
                throw std::runtime_error(fmt::format("Cannot access a VoigtProfile in device memory from CPU..."));
            }
        );
#endif
        T frac_a = (a - a_range.min) / a_step;
        // NOTE(cmo): p suffix for term at idx + 1
        int ia, iap;
        T ta, tap;
        if (frac_a < FP(0.0) || frac_a >= a_range.n) {
            ia = yakl::min(yakl::max(ia, 0), a_range.n);
            iap = ia;
            ta = FP(1.0);
            tap = FP(0.0);
        } else {
            ia = int(frac_a);
            iap = ia + 1;
            tap = frac_a - ia;
            ta = FP(1.0) - tap;
        }
        T frac_v = (v - v_range.min) / v_step;
        int iv, ivp;
        T tv, tvp;
        if (frac_v < FP(0.0) || frac_v >= v_range.n) {
            iv = yakl::min(yakl::max(iv, 0), v_range.n);
            ivp = iv;
            tv = FP(1.0);
            tvp = FP(0.0);
        } else {
            iv = int(frac_v);
            ivp = iv + 1;
            tvp = frac_v - iv;
            tv = FP(1.0) - tvp;
        }
        Voigt_t result = (
            ta * (tv * samples(ia, iv) + tvp * samples(ia, ivp)) +
            tap * (tv * samples(iap, iv) + tvp * samples(iap, ivp))
        );
        return result;
    }
};

#else
#endif