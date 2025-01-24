#if !defined(DEXRT_LOOP_UTILS_HPP)
#define DEXRT_LOOP_UTILS_HPP

#include "Config.hpp"
#include "Types.hpp"

/// All of this is C-style Iterate::Right loops.

template <int N>
struct FlatLoop {
    Kokkos::Array<i32, N> bounds;
    i64 num_iter;

    // FlatLoop(const Kokkos::Array<i32, N>& bounds_) : bounds(bounds_) {
    FlatLoop(i32 b0, i32 b1=0, i32 b2=0, i32 b3=0, i32 b4=0, i32 b5=0) {
        bounds[0] = b0;
        if constexpr (N >= 2) {
            bounds[1] = b1;
        }
        if constexpr (N >= 3) {
            bounds[2] = b2;
        }
        if constexpr (N >= 4) {
            bounds[3] = b3;
        }
        if constexpr (N >= 5) {
            bounds[4] = b4;
        }
        if constexpr (N >= 6) {
            bounds[5] = b5;
        }

        num_iter = 1;
        for (int i = 0; i < N; ++i) {
            KOKKOS_ASSERT(bounds[i] > 0);
            num_iter *= bounds[i];
        }
    }

    FlatLoop(const FlatLoop<N>&) = default;
    FlatLoop(FlatLoop<N>&&) = default;
    FlatLoop<N>& operator=(const FlatLoop<N>&) = default;

    KOKKOS_INLINE_FUNCTION Kokkos::Array<i32, N> unpack(i64 i) const;
};

struct TeamWorkDivision {
    int team_count;
    i64 inner_work_count;
};

template <int N>
struct BalanceLoopArgs {
    const FlatLoop<N>& loop;
    int min_blocks = 256; /// Minimum number of blocks to use, reshuffle from higher indices as needed.
    int max_blocks = 65535; /// Maximum number of blocks, shuffle to lower indices as needed.
};

template <int N>
inline TeamWorkDivision balance_parallel_work_division(const BalanceLoopArgs<N>& args) {
    JasUnpack(args, loop, min_blocks, max_blocks);

    i32 leading_dim = loop.bounds[0];
    i64 trailing_dims = loop.num_iter / leading_dim;
    while (leading_dim < min_blocks) {
        leading_dim *= 2;
        trailing_dims = (trailing_dims + 1) / 2; // ceiling div
    }
    while (leading_dim >= max_blocks) {
        leading_dim = (leading_dim + 1) / 2; // ceiling div
        trailing_dims *= 2;
    }
    return TeamWorkDivision {
        .team_count = leading_dim,
        .inner_work_count = trailing_dims
    };
}

// template <int CurrentLevel=0, class Lambda, int N, typename... Args>
// KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, N>& arr, Args... args) {
//     i32 i = arr[CurrentLevel];
//     if constexpr (CurrentLevel == N - 1) {
//         closure(args..., i);
//     } else {
//         array_invoke<CurrentLevel+1>(closure, arr, args..., i);
//     }
// }

template <int N, class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, N>& arr);

template <class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, 1>& arr) {
    closure(arr[0]);
}

template <class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, 2>& arr) {
    closure(arr[0], arr[1]);
}

template <class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, 3>& arr) {
    closure(arr[0], arr[1], arr[2]);
}

template <class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, 4>& arr) {
    closure(arr[0], arr[1], arr[2], arr[3]);
}

template <class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, 5>& arr) {
    closure(arr[0], arr[1], arr[2], arr[3], arr[4]);
}

template <class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, 6>& arr) {
    closure(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
}


template <class ExecutionSpace=Kokkos::DefaultExecutionSpace, int N, class Lambda>
inline void dex_parallel_for(const std::string& name, const FlatLoop<N>& loop, const Lambda& closure) {
    static_assert(N < 7, "Flat loops only supported for 1 <= N <= 6");
    if constexpr (N == 1) {
        Kokkos::parallel_for(
            name,
            Kokkos::RangePolicy<ExecutionSpace>(0, loop.bounds[0]),
            closure
        );
    } else {
        auto work_div = balance_parallel_work_division(BalanceLoopArgs<N>{
            .loop = loop
        });
        Kokkos::parallel_for(
            name,
            Kokkos::TeamPolicy<ExecutionSpace>(work_div.team_count, Kokkos::AUTO()),
            KOKKOS_LAMBDA (const Kokkos::TeamPolicy<ExecutionSpace>::member_type& team) {
                i64 i = team.league_rank() * work_div.inner_work_count;
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(
                        team,
                        work_div.inner_work_count
                    ),
                    [&] (int j) {
                        i64 idx = i + j;
                        if (idx >= loop.num_iter) {
                            return;
                        }
                        auto idxs = loop.unpack(idx);
                        array_invoke(closure, idxs);
                    }
                );
            }
        );
    }
}

template <>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<i32, 1> FlatLoop<1>::unpack(i64 i) const {
    return {i32(i)};
}

template <>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<i32, 2> FlatLoop<2>::unpack(i64 i) const {
    i32 idx0 = i32(i / bounds[1]);
    i32 idx1 = i32(i - bounds[1] * idx0);
    return {idx0, idx1};
}

template <>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<i32, 3> FlatLoop<3>::unpack(i64 i) const {
    i64 dim_prod = bounds[1] * bounds[2];
    i32 idx0 = i32(i / dim_prod);
    i64 offset = idx0 * dim_prod;
    dim_prod = bounds[2];
    i32 idx1 = i32((i - offset) / dim_prod);
    offset += idx1 * dim_prod;
    i32 idx2 = i32(i - offset);
    return {idx0, idx1, idx2};
}

template <>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<i32, 4> FlatLoop<4>::unpack(i64 i) const {
    i64 dim_prod = bounds[1] * bounds[2] * bounds[3];
    i32 idx0 = i32(i / dim_prod);

    i64 offset = idx0 * dim_prod;
    dim_prod = bounds[2] * bounds[3];
    i32 idx1 = i32((i - offset) / dim_prod);

    offset += idx1 * dim_prod;
    dim_prod = bounds[3];
    i32 idx2 = i32((i - offset) / dim_prod);

    offset += idx2 * dim_prod;
    i32 idx3 = i32(i - offset);
    return {idx0, idx1, idx2, idx3};
}

template <>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<i32, 5> FlatLoop<5>::unpack(i64 i) const {
    i64 dim_prod = bounds[1] * bounds[2] * bounds[3] * bounds[4];
    i32 idx0 = i32(i / dim_prod);

    i64 offset = idx0 * dim_prod;
    dim_prod = bounds[2] * bounds[3] * bounds[4];
    i32 idx1 = i32((i - offset) / dim_prod);

    offset += idx1 * dim_prod;
    dim_prod = bounds[3] * bounds[4];
    i32 idx2 = i32((i - offset) / dim_prod);

    offset += idx2 * dim_prod;
    dim_prod = bounds[4];
    i32 idx3 = i32((i - offset) / dim_prod);

    offset += idx3 * dim_prod;
    i32 idx4 = i32(i - offset);
    return {idx0, idx1, idx2, idx3, idx4};
}

template <>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<i32, 6> FlatLoop<6>::unpack(i64 i) const {
    i64 dim_prod = bounds[1] * bounds[2] * bounds[3] * bounds[4] * bounds[5];
    i32 idx0 = i32(i / dim_prod);

    i64 offset = idx0 * dim_prod;
    dim_prod = bounds[2] * bounds[3] * bounds[4] * bounds[5];
    i32 idx1 = i32((i - offset) / dim_prod);

    offset += idx1 * dim_prod;
    dim_prod = bounds[3] * bounds[4] * bounds[5];
    i32 idx2 = i32((i - offset) / dim_prod);

    offset += idx2 * dim_prod;
    dim_prod = bounds[4] * bounds[5];
    i32 idx3 = i32((i - offset) / dim_prod);

    offset += idx3 * dim_prod;
    dim_prod = bounds[5];
    i32 idx4 = i32((i - offset) / dim_prod);

    offset += idx4 * dim_prod;
    i32 idx5 = i32(i - offset);
    return {idx0, idx1, idx2, idx3, idx4, idx5};
}

#else
#endif