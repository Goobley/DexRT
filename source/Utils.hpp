#if !defined(DEXRT_UTILS_HPP)
#define DEXRT_UTILS_HPP
#include "Config.hpp"
#include "Types.hpp"

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

YAKL_INLINE void mipmap_arr(const FpConst3d& arr, const Fp3d& result, int factor, int x, int y) {
    fp_t weight = FP(1.0) / fp_t(1 << (2 * factor));
    int scale = (1 << factor);
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> temp(FP(0.0));
    for (int off_x = 0; off_x < scale; ++off_x) {
        for (int off_y = 0; off_y < scale; ++off_y) {
            for (int w = 0; w < arr.extent(2); ++w) {
                temp(w) += weight * arr(x * scale + off_x, y * scale + off_y, w);
            }
        }
    }
    for (int w = 0; w < result.extent(2); ++w) {
        result(x, y, w) = temp(w);
    }
}

/** Upper bound on YAKL arrays, returns index rather than iterator.
 * 
*/
template <typename T, int mem_space=yakl::memDevice>
YAKL_INLINE int upper_bound(const yakl::Array<const T, 1, mem_space>& x, T value) {
    int count = x.extent(0);
    int step;
    const T* first = &x(0);
    const T* it;
 
    while (count > 0)
    {
        it = first; 
        step = count / 2; 
        it += step;
 
        if (!(value < *it)) {
            // target in right sub-array
            first = ++it;
            count -= step + 1;
        } else {
            // target in left sub-array
            count = step;
        }
    }
    return first - &x(0);
}

/** Linearly interpolate a sample (at alpha) from array y on grid x. Assumes x is positive sorted.
 * Clamps on ends.
*/
template <typename T=fp_t, int mem_space=yakl::memDevice>
YAKL_INLINE T interp(
    T alpha,
    const yakl::Array<T const, 1, mem_space>& x, 
    const yakl::Array<T const, 1, mem_space>& y
) {
    if (alpha <= x(0)) {
        return y(0);
    } else if (alpha >= x(x.extent(0)-1)) {
        return y(y.extent(0)-1);
    }

    // NOTE(cmo): We know from the previous checks that idxp is in [1,
    // x.extent(0)-1] because our sample is guaranteed inside the grid. This is
    // the upper bound of the linear range.
    int idxp = upper_bound(x, alpha);
    int idx = idxp - 1;

    T t = (x(idxp) - alpha) / (x(idxp) - x(idx));
    return t * y(idx) + (FP(1.0) - t) * y(idxp);
}

template <typename T=fp_t, int mem_space=yakl::memDevice>
YAKL_INLINE T interp(
    T alpha,
    const yakl::Array<T, 1, mem_space>& x, 
    const yakl::Array<T, 1, mem_space>& y
) {
    // NOTE(cmo): The optimiser should eat this up
    yakl::Array<T const, 1, mem_space> xx(x);
    yakl::Array<T const, 1, mem_space> yy(y);
    return interp(alpha, xx, yy);
}

#else
#endif