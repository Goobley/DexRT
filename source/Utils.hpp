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


#else
#endif