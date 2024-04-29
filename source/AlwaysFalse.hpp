#if !defined(DEXRT_ALWAYS_FALSE_HPP)
#define DEXRT_ALWAYS_FALSE_HPP
#include <type_traits>

// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html

/**
 * Can be used in as static_assert for a consistent false as `AlwaysFalse<T>::value`.
*/
template <typename T>
struct AlwaysFalse : std::false_type {
};


#else
#endif