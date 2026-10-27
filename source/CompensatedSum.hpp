#if !defined(DEXRT_COMPENSATED_SUM_HPP)
#define DEXRT_COMPENSATED_SUM_HPP

#include "Types.hpp"
#include <type_traits>

/// Compensated summation: the rounding error of each addition, from Knuth's
/// TwoSum, is accumulated separately in error. Follows https://uwplse.org/2025/08/04/two-sum.html
/// or Alg 4.4 or Handbook of Floating-Point Arithmetic, Muller et al, 2018, Birkhauser-Charm
/// If we compile with fp_t = f64, this is replaced with simple f64 accumulation.
template <typename T, bool Compensate = std::is_same_v<T, float>>
struct CompensatedSumT {
    T sum = T(0.0);
    T error = T(0.0);

    /// Add b in place.
    YAKL_INLINE CompensatedSumT& add(T b) {
        const T s = sum + b;
        const T aa = s - b;
        const T bb = s - aa;
        const T delta_a = sum - aa;
        const T delta_b = b - bb;
        error += delta_a + delta_b;
        sum = s;
        return *this;
    }

    /// Subtract b in place.
    YAKL_INLINE CompensatedSumT& subtract(T b) {
        return add(-b);
    }

    /// The total plus b, leaving this unchanged.
    YAKL_INLINE T plus(T b) const {
        CompensatedSumT result = *this;
        return result.add(b).value();
    }

    /// The total minus b, leaving this unchanged.
    YAKL_INLINE T minus(T b) const {
        CompensatedSumT result = *this;
        return result.subtract(b).value();
    }

    /// The total, rounded once.
    YAKL_INLINE T value() const {
        return sum + error;
    }
};

template <typename T>
struct CompensatedSumT<T, false> {
    T sum = T(0.0);

    YAKL_INLINE CompensatedSumT& add(T b) {
        sum += b;
        return *this;
    }

    YAKL_INLINE CompensatedSumT& subtract(T b) {
        sum -= b;
        return *this;
    }

    YAKL_INLINE T plus(T b) const {
        return sum + b;
    }

    YAKL_INLINE T minus(T b) const {
        return sum - b;
    }

    YAKL_INLINE T value() const {
        return sum;
    }
};

typedef CompensatedSumT<fp_t> CompensatedSum;

#else
#endif
