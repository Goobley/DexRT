#if !defined(DEXRT_VOIGT_HPP)
#define DEXRT_VOIGT_HPP

#include "Types.hpp"
#include "MortonCodes.hpp"
#include "Utils.hpp"
#include <fmt/core.h>
#include <bit>
#include <limits>

#define FPT(X) T(FP(X))
// NOTE(cmo): The complex in Kokkos is much slower because of the branch in operator/.
#if 0
template <typename T>
using DexComplex = Kokkos::complex<T>;
#else
// NOTE(cmo): This version based on thrust.
template <typename T>
struct alignas(2 * sizeof(T)) DexComplex {
  static_assert(std::is_floating_point_v<T> &&
                    std::is_same_v<T, std::remove_cv_t<T>>,
                "DexComplex can only be instantiated for a cv-unqualified "
                "floating point type");

    DexComplex() noexcept = default;
    DexComplex(const DexComplex&) noexcept = default;
    DexComplex& operator=(const DexComplex&) noexcept = default;
    KOKKOS_FORCEINLINE_FUNCTION constexpr DexComplex(const T& re, const T& im) noexcept : re_(re), im_(im) {}
    KOKKOS_FORCEINLINE_FUNCTION constexpr T& imag() noexcept {return im_; }
    KOKKOS_FORCEINLINE_FUNCTION constexpr T& real() noexcept {return re_; }
    KOKKOS_FORCEINLINE_FUNCTION constexpr T imag() const noexcept {return im_; }
    KOKKOS_FORCEINLINE_FUNCTION constexpr T real() const noexcept {return re_; }

    private:
        T re_;
        T im_;
};

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator*(const DexComplex<T>& x, const DexComplex<T>& y) {
    return DexComplex(
        x.real() * y.real() - x.imag() * y.imag(),
        x.real() * y.imag() + x.imag() * y.real()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator*(const DexComplex<T>& x, const T& y) {
    return DexComplex(
        y * x.real(),
        y * x.imag()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator*(const T& y, const DexComplex<T>& x) {
    return DexComplex<T>(
        y * x.real(),
        y * x.imag()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator/(const DexComplex<T>& x, const T& y) {
    return DexComplex<T>(x.real() / y, x.imag() / y);
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator/(const DexComplex<T>& x, const DexComplex<T>& y) {
    // Kokkos::complex<T> result = Kokkos::complex<T>(x.real(), x.imag()) / Kokkos::complex<T>(y.real(), y.imag());
    // return DexComplex<T>(result.real(), result.imag());
    // Based on thrust implementation (Apache-2)
    using std::abs;
    T s = abs(y.real()) + abs(y.imag());

    T oos = FPT(1.0) / s;

    const T ars = x.real() * oos;
    const T ais = x.imag() * oos;
    const T brs = y.real() * oos;
    const T bis = y.imag() * oos;

    s = (brs * brs) + (bis * bis);
    oos = FPT(1.0) / s;
    DexComplex<T> result(
        ((ars * brs) + (ais * bis)) * oos,
        ((ais * brs) - (ars * bis)) * oos
    );
    return result;
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator/(const T& x, const DexComplex<T>& y) {
    return DexComplex<T>(x, FPT(0.0)) / y;
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator+(const DexComplex<T>& x, const DexComplex<T>& y) {
    return DexComplex<T>(
        x.real() + y.real(),
        x.imag() + y.imag()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator+(const T& x, const DexComplex<T>& y) {
    return DexComplex<T>(
        x + y.real(),
        y.imag()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator+(const DexComplex<T>& x, const T& y) {
    return DexComplex<T>(
        x.real() + y,
        x.imag()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator-(const DexComplex<T>& x, const DexComplex<T>& y) {
    return DexComplex<T>(
        x.real() - y.real(),
        x.imag() - y.imag()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator-(const T& x, const DexComplex<T>& y) {
    return DexComplex<T>(
        x - y.real(),
        -y.imag()
    );
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION DexComplex<T> operator-(const DexComplex<T>& x, const T& y) {
    return DexComplex<T>(
        x.real() - y,
        x.imag()
    );
}
#endif


namespace DexVoigtDetail {
    template <template<typename> class Complex, typename T>
    YAKL_INLINE Complex<T> cexp(const Complex<T>& x) {
        using std::exp, std::cos, std::sin;
        return exp(x.real()) * Complex<T>(cos(x.imag()), sin(x.imag()));
    }
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
    // NOTE(cmo): The accuracy seemed a bit low in regions II and III
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
    // NOTE(cmo): Looks good and pretty fast!
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

// NOTE(cmo): The lookup table is a win on CPU, where the Voigt evaluation is a
// large share of the runtime; on GPU the direct evaluation is cheap enough.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
constexpr bool DEX_VOIGT_LUT = false;
#define DEX_HOST_SIMD
#else
constexpr bool DEX_VOIGT_LUT = true;
/// Vectorise a loop on host backends.
#define DEX_HOST_SIMD _Pragma("omp simd")
#endif

/// Real Voigt lookup table bounds. a is sampled at 0, then from 2^a_min_log2
/// up to a_max at the floats with only the top a_mantissa_bits of mantissa set
/// (2^e (1 + m / 2^bits), so the index comes straight from a's bits); |v|
/// uniformly up to v_max. Max relative error ~4e-4 vs wei24. For a + |v| >=
/// v_max wei24's Region I (a single rational term) is used. a_max >= v_max, so
/// the lookup is branch-free (vectorises).
struct VoigtLutParams {
    i32 a_min_log2 = -14;
    f64 a_max = 16.0;
    i32 a_mantissa_bits = 4;
    f64 v_max = 15.0;
    f64 v_step = 0.005;
};

template <typename T=fp_t, bool Complex=false, int mem_space=yakl::memDevice, bool USE_LUT=DEX_VOIGT_LUT && !Complex>
struct VoigtProfile {
    /// Voigt function: a bilinear lookup table (if USE_LUT = true, real only),
    /// otherwise humlicek_wei24_voigt direct evaluation.
    typedef std::conditional_t<Complex, DexComplex<T>, T> Voigt_t;
    typedef std::conditional_t<sizeof(T) == 4, u32, u64> Bits_t;
    static constexpr i32 MANTISSA_DIGITS = std::numeric_limits<T>::digits - 1;
    static_assert(!(USE_LUT && Complex), "The Voigt LUT is only for the real profile");
    T a_min;
    i32 a_shift;
    Bits_t a_key_min;
    Bits_t a_key_last; /// key of the last lower node
    T v_max;
    T inv_v_step;
    T v_frac_last; /// v / v_step of the last lower node
    yakl::Array<T const, 2, mem_space> samples; /// [a, v]

    VoigtProfile() {};

    /// Construct (and for the LUT, fill the table).
    static VoigtProfile make(const VoigtLutParams& p = VoigtLutParams{}) {
        VoigtProfile result;
        if constexpr (USE_LUT) {
            result.fill(p);
        }
        return result;
    }

    /// a at node ia (> 0).
    YAKL_INLINE T a_node(Bits_t key) const {
        return std::bit_cast<T>(Bits_t(key << a_shift));
    }

    void fill(const VoigtLutParams& p) {
        if (p.a_max < p.v_max) {
            throw std::runtime_error("Voigt LUT: a_max must be >= v_max");
        }
        a_min = std::ldexp(T(1.0), p.a_min_log2);
        a_shift = MANTISSA_DIGITS - p.a_mantissa_bits;
        a_key_min = std::bit_cast<Bits_t>(a_min) >> a_shift;
        const Bits_t a_key_max = std::bit_cast<Bits_t>(T(p.a_max)) >> a_shift;
        a_key_last = a_key_max;
        const i32 num_a = i32(a_key_max - a_key_min) + 3;
        const i32 num_v = i32(std::ceil(p.v_max / p.v_step)) + 2;
        v_frac_last = T(num_v - 2);
        yakl::Array<T, 2, yakl::memHost> s_host("Voigt LUT", num_a, num_v);
        for (int ia = 0; ia < num_a; ++ia) {
            const f64 a = (ia == 0) ? 0.0 : f64(a_node(a_key_min + ia - 1));
            for (int iv = 0; iv < num_v; ++iv) {
                s_host(ia, iv) = T(humlicek_wei24_voigt<f64>(a, iv * p.v_step).real());
            }
        }
        v_max = T(p.v_max);
        inv_v_step = T(1.0 / p.v_step);
        if constexpr (mem_space == yakl::memDevice) {
            samples = s_host.createDeviceCopy();
        } else {
            samples = s_host;
        }
    }

    /// The table lookup (USE_LUT) of H(a, v) for N pairs (v >= 0) into out.
    /// Vector passes for the indices/weights and the interpolation, around a
    /// scalar fetch (gathers are slow, and disabled, on recent Intel); the
    /// members are copied to locals so nothing is loaded conditionally.
    template <int N>
    YAKL_INLINE void lookup(const T* a_in, const T* v_in, T* out, int n = N) const {
        const T* __restrict table = samples.data();
        const i32 stride = samples.extent(1);
        const T a_min_ = a_min;
        const i32 a_shift_ = a_shift;
        const Bits_t a_key_min_ = a_key_min;
        const Bits_t a_key_last_ = a_key_last;
        const T v_max_ = v_max;
        const T inv_v_step_ = inv_v_step;
        const T v_frac_last_ = v_frac_last;
        i32 offset[N];
        T ta[N];
        T tv[N];
        T region_1[N];
        i32 in_table[N];
        DEX_HOST_SIMD
        for (int i = 0; i < n; ++i) {
            const T a = a_in[i];
            const T v = v_in[i];
            // NOTE(cmo): Both the table and Region I are evaluated, then
            // selected; indices are clamped for the Region I lanes.
            const T a2 = a * a;
            const T v2 = v * v;
            const T dr = T(0.5) + a2 - v2;
            region_1[i] = T(0.5641896) * a * (T(0.5) + a2 + v2) / (dr * dr + T(4.0) * a2 * v2);
            in_table[i] = (a + v < v_max_) ? 1 : 0;

            // NOTE(cmo): Node 0 is a = 0, then node 1 + k is key a_key_min + k.
            const bool small = a < a_min_;
            const Bits_t raw = std::bit_cast<Bits_t>(a) >> a_shift_;
            const Bits_t key = small ? Bits_t(a_key_min_ - 1) : (raw < a_key_last_ ? raw : a_key_last_);
            const T a0 = small ? T(0.0) : std::bit_cast<T>(Bits_t(key << a_shift_));
            const T a1 = std::bit_cast<T>(Bits_t((key + 1) << a_shift_));
            ta[i] = (a - a0) / (a1 - a0);
            const T fv_raw = v * inv_v_step_;
            const T fv = (fv_raw < v_frac_last_) ? fv_raw : v_frac_last_;
            const i32 iv = i32(fv);
            tv[i] = fv - T(iv);
            offset[i] = (1 + i32(key - a_key_min_)) * stride + iv;
        }
        T t00[N];
        T t01[N];
        T t10[N];
        T t11[N];
        for (int i = 0; i < n; ++i) {
            const T* row = table + offset[i];
            t00[i] = row[0];
            t01[i] = row[1];
            t10[i] = row[stride];
            t11[i] = row[stride + 1];
        }
        DEX_HOST_SIMD
        for (int i = 0; i < n; ++i) {
            const T lo = t00[i] + tv[i] * (t01[i] - t00[i]);
            const T hi = t10[i] + tv[i] * (t11[i] - t10[i]);
            const T result = lo + ta[i] * (hi - lo);
            out[i] = (in_table[i] != 0) ? result : region_1[i];
        }
    }

    /// operator() for N (a, v) pairs into out (vectorises on CPU).
    template <int N>
    YAKL_INLINE void eval(const T* a, const T* v, Voigt_t* out, int n = N) const {
        if constexpr (USE_LUT) {
            T v_abs[N];
            for (int i = 0; i < n; ++i) {
                v_abs[i] = std::abs(v[i]);
            }
            lookup<N>(a, v_abs, out, n);
        } else {
            for (int i = 0; i < n; ++i) {
                out[i] = (*this)(a[i], v[i]);
            }
        }
    }

    YAKL_INLINE Voigt_t operator()(T a, T v) const {
        if constexpr (!Complex) {
            v = std::abs(v);
        }
        using result_t = DexVoigtDetail::ComplexOrReal<T, Complex>;
        if constexpr (USE_LUT) {
            T result;
            lookup<1>(&a, &v, &result);
            return result;
        }
        return result_t::value(humlicek_wei24_voigt(a, v));
    }
};

#else
#endif