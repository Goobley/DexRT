#if !defined(DEXRT_UTILS_HPP)
#define DEXRT_UTILS_HPP
#include "Config.hpp"

template <typename T>
YAKL_INLINE bool approx_equal(T a, T b, T eps)
{
    // https://floating-point-gui.de/errors/comparison/

    const T absA = std::abs(a);
    const T absB = std::abs(b);
    const T diff = std::abs(a - b);

    if (a == b)
    {
        return true;
    }
    else if (a == 0.0 || b == 0.0 || (absA + absB < std::numeric_limits<T>::min()))
    {
        // NOTE(cmo): min in std::numeric_limits is the minimum normalised value.
        return diff < (eps * std::numeric_limits<T>::min());
    }
    else
    {
        return diff / min(absA + absB, std::numeric_limits<T>::max()) < eps;
    }
}

template <typename T>
YAKL_INLINE T sign(T t) {
    return std::copysign(T(1.0), t);
}

using namespace yakl::componentwise;
template <typename T>
YAKL_INLINE constexpr auto square(T t) -> decltype(t * t) {
    return t * t;
}

template <typename T>
YAKL_INLINE constexpr auto cube(T t) -> decltype(t * t * t) {
    return t * t * t;
}

struct Mipmapper {
    const FpConst3d& arr;
    const Fp3d& result;
    int factor;

    Mipmapper(
        const FpConst3d& array,
        const Fp3d& result_storage,
        int mip_factor
    ) : arr(array),
        result(result_storage),
        factor(mip_factor)
    {}

    YAKL_INLINE void operator()(int x, int y) {
        fp_t weight = FP(1.0) / fp_t(1 << (2 * factor));
        int scale = (1 << factor);
        yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> temp(FP(0.0));
        for (int off_x = 0; off_x < scale; ++off_x) {
            for (int off_y = 0; off_y < scale; ++off_y) {
                for (int w = 0; w < NUM_WAVELENGTHS; ++w) {
                    temp(w) += weight * arr(x * scale + off_x, y * scale + off_y, w);
                }
            }
        }
        for (int w = 0; w < NUM_WAVELENGTHS; ++w) {
            result(x, y, w) = temp(w);
        }
    }
};

#else
#endif