#if !defined(DEXRT_VOIGT_HPP)
#define DEXRT_VOIGT_HPP

#include "Types.hpp"
#include "MortonCodes.hpp"
#include "Utils.hpp"
#include <fmt/core.h>

// #if defined(YAKL_ARCH_CUDA)
// #include <cuda/std/complex>
// template <typename T>
// using DexComplex = cuda::std::complex<T>;
// #elif defined(YAKL_ARCH_HIP)
#if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP)
#include <thrust/complex.h>
template <typename T>
using DexComplex = thrust::complex<T>;
#elif defined(YAKL_ARCH_SYCL)
#define SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
template <typename T>
using DexComplex = sycl::_V1::ext::oneapi::experimental::complex<T>;
#else
#include <complex>
template <typename T>
using DexComplex = std::complex<T>;
#endif

namespace DexVoigtDetail {
template <typename T>
YAKL_INLINE T cexp(const T& x) {
// #if defined(YAKL_ARCH_CUDA)
    // return cuda::std::exp(x);
// #elif defined(YAKL_ARCH_HIP)
#if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP)
    return thrust::exp(x);
#elif defined(YAKL_ARCH_SYCL)
    return sycl::_V1::ext::oneapi::experimental::exp(x);
#else
    return std::exp(x);
#endif
}
}

#define FPT(X) T(FP(X))
template <typename T=fp_t>
YAKL_INLINE T abrarov_quine_voigt(T a, T v) {
    // Not convinced this is working correctly.
    // https://www.ccsenet.org/journal/index.php/jmr/article/view/62103
    // Real only!
    constexpr int num_terms = 12;
    // First column
    constexpr T A[num_terms] = {
        FPT(2.307372754308023e-001),
        FPT(7.760531995854886e-001),
        FPT(4.235506885098250e-002),
        FPT(-2.340509255269456e-001),
        FPT(-4.557204758971222e-002),
        FPT(5.043797125559205e-003),
        FPT(1.180179737805654e-003),
        FPT(1.754770213650354e-005),
        FPT(-3.325020499631893e-006),
        FPT(-9.375402319079375e-008),
        FPT(8.034651067438904e-010),
        FPT(3.355455275373310e-011)
    };

    // Second column
    constexpr T B[num_terms] = {
        FPT(4.989787261063716e-002),
        FPT(4.490808534957343e-001),
        FPT(1.247446815265929e+000),
        FPT(2.444995757921221e+000),
        FPT(4.041727681461610e+000),
        FPT(6.037642585887094e+000),
        FPT(8.432740471197681e+000),
        FPT(1.122702133739336e+001),
        FPT(1.442048518447414e+001),
        FPT(1.801313201244001e+001),
        FPT(2.200496182129099e+001),
        FPT(2.639597461102705e+001)
    };

    // Third column
    constexpr T C[num_terms] = {
        FPT(1.464495070025765e+000),
        FPT(-3.230894193031240e-001),
        FPT(-5.397724160374686e-001),
        FPT(-6.547649406082363e-002),
        FPT(2.411056013969393e-002),
        FPT(4.001198804719684e-003),
        FPT(-5.387428751666454e-005),
        FPT(-2.451992671326258e-005),
        FPT(-5.400164289522879e-007),
        FPT(1.771556420016014e-008),
        FPT(4.940360170163906e-010),
        FPT(5.674096644030151e-014)
    };

    constexpr T varsigma = FPT(2.75);
    v = std::abs(v) + FPT(0.5) * varsigma;

    const T x1 = square(v) - square(a);
    const T x2 = square(a) + square(v);
    const T x3 = square(x2);

    T result = FPT(0.0);
    #pragma unroll
    for (int i = 0; i < num_terms; ++i) {
        result += (
            (A[i] * (B[i] + x1) + C[i] * v * (B[i] + x2))
            / (square(B[i]) + 2 * B[i] * x1 + x3)
        );
    }
    return result;
}

template <typename T=fp_t>
YAKL_INLINE DexComplex<T> humlicek_voigt(T a, T v) {
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
        u*(FPT(219.031) - u*(FPT(35.7668) - u*(FPT(1.320522) - u*FPT(0.5641896))))))) /
        (FPT(32066.6) - u*(FPT(24322.84) - u*(FPT(9022.228) - u*(FPT(2186.181) -
        u*(FPT(364.2191) - u*(FPT(61.57037) - u*(FPT(1.841439) - u))))))));
    }
}

// https://academic.oup.com/mnras/article/479/3/3068/5045263#equ10
template <typename T=fp_t>
YAKL_INLINE DexComplex<T> humlicek_zpf16_voigt(T a, T v) {
    // !   Humlicek zpf16 constants
    const DexComplex<T> ac[16] = {
        {FPT(41445.0374210222), FPT(0.0)},
        {FPT(0.0), FPT(-136631.072925829)},
        {FPT(-191726.143960199), FPT(0.0)},
        {FPT(0.0), FPT(268628.568621291)},
        {FPT(173247.907201704),  FPT(0.0)},
        {FPT(0.0), FPT(-179862.56759178)},
        {FPT(-63310.0020563537), FPT(0.0)},
        {FPT(0.0), FPT(56893.7798630723)},
        {FPT(11256.4939105413),  FPT(0.0)},
        {FPT(0.0), FPT(-9362.62673144278)},
        {FPT(-1018.67334277366), FPT(0.0)},
        {FPT(0.0), FPT(810.629101627698)},
        {FPT(44.5707404545965), FPT(0.0)},
        {FPT(0.0), FPT(-34.5401929182016)},
        {FPT(-0.740120821385939), FPT(0.0)},
        {FPT(0.0), FPT(0.564189583547714)}
    };
    constexpr T b[8] = {
        FPT(7918.06640624997),
        FPT(-126689.0625),
        FPT(295607.8125),
        FPT(-236486.25),
        FPT(84459.375),
        FPT(-15015.0),
        FPT(1365.0),
        FPT(-60.0)
    };

    T s = std::abs(v) + a;
    if (s >= FPT(15.0)) {
        // Region I
        DexComplex<T> z(a, -v);
        return (z * FPT(0.5641896)) / (FPT(0.5) + z * z);
    } else {
        DexComplex<T> z(v, a + FPT(1.31183));
        auto zz = square(z);
        auto numer = ((((((ac[15]*z+ac[14])*z+ac[13])*z+ac[12])*z+ac[11])*z+ac[10])*z+ac[9])*z+ac[8];
        numer = ((((((((numer*z+ac[7])*z+ac[6])*z+ac[5])*z+ac[4])*z+ac[3])*z+ac[2])*z+ac[1])*z+ac[0]);
        auto denom = b[0]+(b[1]+(b[2]+(b[3]+(b[4]+(b[5]+(b[6]+b[7]*zz)*zz)*zz)*zz)*zz)*zz)*zz;
        return numer / denom;
    }
}

// https://academic.oup.com/mnras/article/479/3/3068/5045263#equ10
template <typename T=fp_t>
YAKL_INLINE DexComplex<T> humlicek_wei24_voigt(T a, T v) {
    constexpr T l = FPT(4.1195342878142354); // ! l=sqrt(n/sqrt(2.))  ! L = 2**(-1/4) * N**(1/2)
    constexpr T ac[24] = {
        FPT(-1.5137461654527820e-10), FPT(4.9048215867870488e-09),
        FPT(1.3310461806370372e-09),  FPT(-3.0082822811202271e-08),
        FPT(-1.9122258522976932e-08), FPT(1.8738343486619108e-07),
        FPT(2.5682641346701115e-07),  FPT(-1.0856475790698251e-06),
        FPT(-3.0388931839840047e-06), FPT(4.1394617248575527e-06),
        FPT(3.0471066083243790e-05),  FPT(2.4331415462641969e-05),
        FPT(-2.0748431511424456e-04), FPT(-7.8166429956142650e-04),
        FPT(-4.9364269012806686e-04), FPT(6.2150063629501763e-03),
        FPT(3.3723366855316413e-02),  FPT(1.0838723484566792e-01),
        FPT(2.6549639598807689e-01),  FPT(5.3611395357291292e-01),
        FPT(9.2570871385886788e-01),  FPT(1.3948196733791203e+00),
        FPT(1.8562864992055408e+00),  FPT(2.1978589365315417e+00)
    };

    T s = std::abs(v) + a;
    if (s >= FPT(15.0)) {
        // Region I
        DexComplex<T> z(a, -v);
        return (z * FPT(0.5641896)) / (FPT(0.5) + z * z);
    } else {
        DexComplex<T> recLmZ = FPT(1.0) / DexComplex<T>(l + a, -v);
        DexComplex<T> t = DexComplex<T>(l - a, v) * recLmZ;
        DexComplex<T> result = recLmZ * (FPT(0.5641896) + FPT(2.0) * recLmZ * (
            ac[23]+(ac[22]+(ac[21]+(ac[20]+(ac[19]+(ac[18]+(ac[17]+(ac[16]+(ac[15]+(ac[14]+(ac[13]+(ac[12]+(ac[11]+(ac[10]+(ac[9]+(ac[8]+
            (ac[7]+(ac[6]+(ac[5]+(ac[4]+(ac[3]+(ac[2]+(ac[1]+ac[0]*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t
        ));
        return result;
    }
}


#undef FPT

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

constexpr bool VOIGT_HYPERBLOCK = true;
constexpr bool VOIGT_MORTON = false;
constexpr i32 VOIGT_TILESIZE = 32;
template <typename T=fp_t, bool Complex=false, int mem_space=yakl::memDevice>
struct VoigtProfile {
    /// Voigt function interpolator
    /// Assumes a and v are linearly interpolated on linspaces, so
    /// these are implicit, rather than stored.
    /// For non-complex voigt, expects min v to be 0, as it is symmetric around this.
    /// Return value clamped to domain if sampled outside.
    using Linspace = DexVoigtDetail::Linspace<T>;
    typedef std::conditional_t<Complex, DexComplex<T>, T> Voigt_t;
    Linspace a_range;
    T a_step;
    Linspace v_range;
    T v_step;
    yakl::Array<Voigt_t const, 2, mem_space> samples;
    i32 num_v_tiles;

    VoigtProfile() {};
    VoigtProfile(Linspace _a_range, Linspace _v_range) :
        a_range(_a_range),
        v_range(_v_range)
    {
        a_step = (a_range.max - a_range.min) / T(a_range.n - 1);
        v_step = (v_range.max - v_range.min) / T(v_range.n - 1);
        num_v_tiles = v_range.n / VOIGT_TILESIZE;
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
        const i32 num_v_tiles = this->num_v_tiles;
        using result_t = DexVoigtDetail::ComplexOrReal<T, Complex>;
        if constexpr (mem_space == yakl::memDevice) {
            parallel_for(
                "compute voigt",
                SimpleBounds<2>(a_range.n, v_range.n),
                YAKL_LAMBDA (int ia, int iv) {
                    constexpr i32 TILESIZE = VOIGT_TILESIZE;
                    const i32 num_tiles = num_v_tiles;
                    const auto sample = voigt_sample(ia, iv);
                    mut_samples(ia, iv);
                    if constexpr (VOIGT_HYPERBLOCK) {
                        i32 iat = ia % TILESIZE;
                        i32 ivt = iv % TILESIZE;
                        i32 tile = (ia / TILESIZE) * num_tiles + iv / TILESIZE;
                        i32 offset;
                        if constexpr (VOIGT_MORTON) {
                            offset = encode_morton_2(Coord2{
                                .x = ivt,
                                .z = iat
                            });
                        } else {
                            offset = iat * TILESIZE + ivt;
                        }
                        i32 idx = tile * square(TILESIZE) + offset;
                        mut_samples.get_data()[idx] = result_t::value(sample);
                    } else {
                        mut_samples(ia, iv) = result_t::value(sample);
                    }
                }
            );
        } else {
            for (int ia = 0; ia < a_range.n; ++ia) {
                for (int iv = 0; iv < v_range.n; ++iv) {
                    if constexpr (VOIGT_HYPERBLOCK) {
                        i32 iat = ia % VOIGT_TILESIZE;
                        i32 ivt = iv % VOIGT_TILESIZE;
                        i32 tile = (ia / VOIGT_TILESIZE) * num_v_tiles + iv / VOIGT_TILESIZE;
                        i32 offset;
                        if constexpr (VOIGT_MORTON) {
                            offset = encode_morton_2(Coord2{
                                .x = ivt,
                                .z = iat
                            });
                        } else {
                            offset = iat * VOIGT_TILESIZE + ivt;
                        }
                        i32 idx = tile * square(VOIGT_TILESIZE) + offset;
                        mut_samples.get_data()[idx] = result_t::value(voigt_sample(ia, iv));
                    } else {
                        mut_samples(ia, iv) = result_t::value(voigt_sample(ia, iv));
                    }
                }
            }
        }
        samples = mut_samples;
    }

    /// Simple clamped bilinear lookup
//     YAKL_INLINE Voigt_t operator()(T a, T v) const {
// #if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
// #ifdef DEXRT_DEBUG
//         YAKL_EXECUTE_ON_HOST_ONLY(
//             if constexpr (mem_space == memDevice) {
//                 yakl::yakl_throw("Cannot access a VoigtProfile in device memory from CPU...");
//             }
//         );
// #endif
// #endif
//         if constexpr (!Complex) {
//             v = std::abs(v);
//         }
//         T frac_a = (a - a_range.min) / a_step;
//         // NOTE(cmo): p suffix for term at idx + 1
//         int ia = int(std::floor(frac_a));
//         int iap;
//         T ta, tap;

//         const i32 max_a = a_range.n - 1;
//         const i32 max_v = v_range.n - 1;
// #ifdef DEXRT_DEBUG
//         if (frac_a < FP(0.0) || frac_a >= (a_range.n - 1)) {
//             ia = yakl::min(yakl::max(ia, 0), a_range.n - 1);
//             iap = ia;
//             ta = FP(1.0);
//             tap = FP(0.0);
//         } else {
//             iap = ia + 1;
//             tap = frac_a - ia;
//             ta = FP(1.0) - tap;
//         }
// #else
//         iap = ia + 1;
//         tap = frac_a - ia;
//         ta = FP(1.0) - tap;
// #endif
//         T frac_v = (v - v_range.min) / v_step;
//         int iv = int(std::floor(frac_v));
//         int ivp;
//         T tv, tvp;
// #ifdef DEXRT_DEBUG
//         if (frac_v < FP(0.0) || frac_v >= (v_range.n - 1)) {
//             iv = yakl::min(yakl::max(iv, 0), v_range.n - 1);
//             ivp = iv;
//             tv = FP(1.0);
//             tvp = FP(0.0);
//         } else {
//             ivp = iv + 1;
//             tvp = frac_v - iv;
//             tv = FP(1.0) - tvp;
//         }
// #else
//         ivp = iv + 1;
//         tvp = frac_v - iv;
//         tv = FP(1.0) - tvp;
// #endif

//         const i32 num_tiles = this->num_v_tiles;
//         auto compute_idx = [num_tiles] (i32 ia, i32 iv) -> i32 {
//             i32 iat = ia % VOIGT_TILESIZE;
//             i32 ivt = iv % VOIGT_TILESIZE;
//             i32 tile = (ia / VOIGT_TILESIZE) * num_tiles + iv / VOIGT_TILESIZE;
//             i32 offset;
//             if constexpr (VOIGT_MORTON) {
//                 offset = encode_morton_2(Coord2{
//                     .x = ivt,
//                     .z = iat
//                 });
//             } else {
//                 offset = iat * VOIGT_TILESIZE + ivt;
//             }
//             i32 idx = tile * square(VOIGT_TILESIZE) + offset;
//             return idx;
//         };

//         auto compute_idx_flat = [max_v] (i32 ia, i32 iv) -> i32 {
//             return ia * max_v + iv;
//         };

//         auto idx2 = [compute_idx, compute_idx_flat] (i32 ia, i32 iv) -> i32 {
//             if constexpr (VOIGT_HYPERBLOCK) {
//                 return compute_idx(ia, iv);
//             } else {
//                 return compute_idx_flat(ia, iv);
//             }
//         };

// #if 0
//         Voigt_t result = (
//             ta * (tv * samples(ia, iv) + tvp * samples(ia, ivp)) +
//             tap * (tv * samples(iap, iv) + tvp * samples(iap, ivp))
//         );
// #else
//         const fp_t* ptr = samples.get_data();
//         Voigt_t result;

//         Voigt_t sample0;
//         Voigt_t sample1;
//         Voigt_t sample2;
//         Voigt_t sample3;

//         if (ia < 0 || ia >= max_a) {
//             ia = std::min(std::max(ia, 0), max_a);
//             if (iv < 0 || iv >=  max_v) {
//                 iv = std::min(std::max(iv, 0), max_v);
//                 sample0 = ptr[idx2(ia, iv)];
//                 sample1 = sample0;
//                 sample2 = sample0;
//                 sample3 = sample0;
//             } else {
//                 sample0 = ptr[idx2(ia, iv)];
//                 sample1 = ptr[idx2(ia, ivp)];
//                 sample2 = sample0;
//                 sample3 = sample1;
//             }
//         } else if (iv < 0 || iv >=  max_v) {
//             iv = std::min(std::max(iv, 0), max_v);
//             sample0 = ptr[idx2(ia, iv)];
//             sample1 = sample0;
//             sample2 = ptr[idx2(iap, iv)];
//             sample3 = sample2;
//         } else {
//             if constexpr (VOIGT_HYPERBLOCK) {
//                 i32 idx_ia = compute_idx(ia, iv);
//                 i32 idx_iap = compute_idx(iap, iv);
//                 sample0 = ptr[idx_ia];
//                 sample1 = ptr[idx_ia + 1];
//                 sample2 = ptr[idx_iap];
//                 sample3 = ptr[idx_iap + 1];
//             } else {
//                 sample0 = ptr[compute_idx_flat(ia, iv)];
//                 sample1 = ptr[compute_idx_flat(ia, ivp)];
//                 sample2 = ptr[compute_idx_flat(iap, iv)];
//                 sample3 = ptr[compute_idx_flat(iap, ivp)];
//             }
//         }

//         result = (
//             ta * (tv * sample0 + tvp * sample1) +
//             tap * (tv * sample2 + tvp * sample3)
//         );
// #endif
//         return result;
//     }

    // YAKL_INLINE Voigt_t operator()(T a, T v) const {
    //     if constexpr (!Complex) {
    //         v = std::abs(v);
    //     }
    //     using result_t = DexVoigtDetail::ComplexOrReal<T, Complex>;
    //     return result_t::value(humlicek_voigt(a, v));
    // }

    // YAKL_INLINE Voigt_t operator()(T a, T v) const {
    //     if constexpr (!Complex) {
    //         v = std::abs(v);
    //     }
    //     using result_t = DexVoigtDetail::ComplexOrReal<T, Complex>;
    //     return result_t::value(humlicek_zpf16_voigt(a, v));
    // }

    // YAKL_INLINE Voigt_t operator()(T a, T v) const {
    //     if constexpr (!Complex) {
    //         v = std::abs(v);
    //     }
    //     return abrarov_quine_voigt(a, v);
    // }

    YAKL_INLINE Voigt_t operator()(T a, T v) const {
        if constexpr (!Complex) {
            v = std::abs(v);
        }
        using result_t = DexVoigtDetail::ComplexOrReal<T, Complex>;
        return result_t::value(humlicek_wei24_voigt(a, v));
    }
};

#else
#endif