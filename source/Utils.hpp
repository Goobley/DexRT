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

/** Upper bound on Kokkos views, returns index rather than iterator.
 *
*/
template <typename ViewType, typename T=ViewType::non_const_value_type>
YAKL_INLINE int upper_bound(const ViewType& x, const T& value) {
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
template <typename ViewType, typename T=ViewType::non_const_value_type>
KOKKOS_INLINE_FUNCTION T interp(
    const T& alpha,
    const ViewType& x,
    const ViewType& y
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

// template <typename T=fp_t, int mem_space=yakl::memDevice>
// YAKL_INLINE T interp(
//     T alpha,
//     const yakl::Array<T, 1, mem_space>& x,
//     const yakl::Array<T, 1, mem_space>& y
// ) {
//     // NOTE(cmo): The optimiser should eat this up
//     // Narrator: It did not (because of the mutex in OpenMP land)
//     yakl::Array<T const, 1, mem_space> xx(x);
//     yakl::Array<T const, 1, mem_space> yy(y);
//     return interp(alpha, xx, yy);
// }

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE
KView<const u16*, mem_space> slice_active_set(const AtomicData<T, mem_space>& atom, int la) {
    // NOTE(cmo): I have no idea why the original slicing (taking
    // &atom.active_lines(start)) as the pointer wasn't working... and was
    // causing "host array being accessed in a device kernel". This seems fine on nvhpc12.1
    const int start = atom.active_lines_start(la);
    const int end = atom.active_lines_end(la);
    auto result = Kokkos::subview(atom.active_lines, std::make_pair(start, end));
    return result;
}

template <typename T=fp_t>
YAKL_INLINE
KView<const u16*> slice_active_cont_set(const AtomicData<T>& atom, int la) {
    const int start = atom.active_cont_start(la);
    const int end = atom.active_cont_end(la);
    auto result = Kokkos::subview(atom.active_cont, std::make_pair(start, end));
    return result;
}

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE
Fp3d slice_pops(const Fp3d& pops, const AtomicData<T, mem_space>& adata, int ia) {
    int start = adata.level_start(ia);
    int num = adata.num_level(ia);
    auto result = Kokkos::subview(pops, std::make_pair(start, start+num), Kokkos::ALL(), Kokkos::ALL());
    return result;
}

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE
Fp2d slice_pops(const Fp2d& pops, const AtomicData<T, mem_space>& adata, int ia) {
    int start = adata.level_start(ia);
    int num = adata.num_level(ia);
    auto result = Kokkos::subview(pops, std::make_pair(start, start+num), Kokkos::ALL());
    return result;
}

template <class ViewType>
auto create_device_copy(const ViewType& arr) -> KView<typename ViewType::data_type, DefaultMemSpace> {
    KView<typename ViewType::non_const_data_type, DefaultMemSpace> result(arr.label(), arr.layout());
    // NOTE(cmo): Directly accessible or layout is the same... no intermediate needed
    if constexpr (
        Kokkos::SpaceAccessibility<typename ViewType::memory_space, DefaultMemSpace>::accessible || std::is_same_v<typename ViewType::array_layout, typename decltype(result)::array_layout>
    ) {
        Kokkos::deep_copy(result, arr);
    // NOTE(cmo): Need intermediate
    } else {
        auto host_temp = Kokkos::create_mirror_view(result);
        Kokkos::deep_copy(host_temp, arr);
        Kokkos::deep_copy(result, host_temp);
    }
    return result;
}

/// Returns a flattened (but unowned) View, i.e. points to the same memory
template <class ViewType>
auto collapse(const ViewType& arr) {
    assert(arr.span_is_contiguous() && "Span must be contiguous to be collapsed");
    return KView<typename ViewType::value_type*, typename ViewType::memory_space>(arr.data(), arr.size());
}

/// Converts from normalised direction vector dir to polar angles [theta, phi] -- theta relative to y
YAKL_INLINE
vec2 dir_to_angs(vec3 dir) {
    using ConstantsFP::pi;

    fp_t sin_theta = std::sqrt(FP(1.0) - square(dir(1)));
    fp_t theta = std::acos(dir(1));
    fp_t phi = std::atan2(dir(2) / sin_theta, dir(0) / sin_theta);
    if (phi < FP(0.0)) {
        phi += FP(2.0) * pi;
    }
    vec2 result;
    result(0) = theta;
    result(1) = phi;
    return result;
}

/// Converts from polar angles [theta, phi] in radians, theta relative to y, to normalised direction vector
YAKL_INLINE
vec3 angs_to_dir(vec2 angs) {
    fp_t cp = std::cos(angs(1));
    fp_t sp = std::sin(angs(1));
    fp_t ct = std::cos(angs(0));
    fp_t st = std::sin(angs(0));
    vec3 d;
    d(0) = cp * st;
    d(1) = ct;
    d(2) = sp * st;
    return d;
}

#else
#endif