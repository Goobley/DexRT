#if !defined(DEXRT_VOIGT_HPP)
#define DEXRT_VOIGT_HPP

#include "Types.hpp"
#include "MortonCodes.hpp"
#include "Utils.hpp"
#include <fmt/core.h>

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
    using std::abs;
    T s = abs(x.real()) + abs(y.imag());

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

template <typename T=fp_t, bool Complex=false, int mem_space=yakl::memDevice, bool USE_LUT=false>
struct VoigtProfile {
    /// Voigt function interpolator (if USE_LUT = true), otherwise using humlicek_wei24_voigt direct evaluation.
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

    VoigtProfile() {};
    VoigtProfile(Linspace _a_range, Linspace _v_range) :
        a_range(_a_range),
        v_range(_v_range)
    {
        if constexpr (!USE_LUT) {
            return;
        }
        a_step = (a_range.max - a_range.min) / T(a_range.n - 1);
        v_step = (v_range.max - v_range.min) / T(v_range.n - 1);
        // NOTE(cmo): Fill table
        compute_samples();
    }


    void compute_samples() {
        if constexpr (!USE_LUT) {
            throw std::runtime_error("No samples to compute: not using LUT for Voigt");
        }
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
                    const auto sample = voigt_sample(ia, iv);
                    mut_samples(ia, iv) = result_t::value(sample);
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
    template <bool InnerUseLut = USE_LUT, std::enable_if_t<InnerUseLut, bool> = true>
    YAKL_INLINE Voigt_t operator()(T a, T v) const {
        if constexpr (!Complex) {
            v = std::abs(v);
        }
        T frac_a = (a - a_range.min) / a_step;
        // NOTE(cmo): p suffix for term at idx + 1
        int ia = int(std::floor(frac_a));
        int iap;
        T ta, tap;

        const i32 max_a = a_range.n - 1;
        const i32 max_v = v_range.n - 1;
        if (frac_a < FP(0.0) || frac_a >= (a_range.n - 1)) {
            ia = std::min(std::max(ia, 0), a_range.n - 1);
            iap = ia;
            ta = FP(1.0);
            tap = FP(0.0);
        } else {
            iap = ia + 1;
            tap = frac_a - ia;
            ta = FP(1.0) - tap;
        }

        T frac_v = (v - v_range.min) / v_step;
        int iv = int(std::floor(frac_v));
        int ivp;
        T tv, tvp;

        if (frac_v < FP(0.0) || frac_v >= (v_range.n - 1)) {
            iv = std::min(std::max(iv, 0), v_range.n - 1);
            ivp = iv;
            tv = FP(1.0);
            tvp = FP(0.0);
        } else {
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

    template <bool InnerUseLut = USE_LUT, std::enable_if_t<!InnerUseLut, bool> = true>
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