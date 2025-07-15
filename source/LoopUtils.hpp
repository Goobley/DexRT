#if !defined(DEXRT_LOOP_UTILS_HPP)
#define DEXRT_LOOP_UTILS_HPP

#include "Config.hpp"
#include "JasPP.hpp"

/// All of this is C-style Iterate::Right loops.

// NOTE(cmo): Define the line below to swap to the default loop being over Kokkos::TeamPolicy
// #define DEX_BALANCE_TEAM_WORK

template <int N>
struct FlatLoop {
    Kokkos::Array<i32, N> bounds;
    i64 num_iter;

    KOKKOS_INLINE_FUNCTION FlatLoop(i32 b0, i32 b1=0, i32 b2=0, i32 b3=0, i32 b4=0, i32 b5=0) {
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

    KOKKOS_INLINE_FUNCTION Kokkos::Array<i32, N> unpack(i64 i) const;
    KOKKOS_INLINE_FUNCTION i32 dim(i32 i) const {
        return bounds[i];
    }
};

struct TeamWorkDivision {
    int team_count;
    i64 inner_work_count;
};

template <int N>
struct BalanceLoopArgs {
    const FlatLoop<N>& loop;
    int min_blocks = 256; /// Minimum number of blocks to use, reshuffle from higher indices as needed.
    int max_blocks = 8192; /// Maximum number of blocks, shuffle to lower indices as needed.
};

template <int N>
inline TeamWorkDivision balance_parallel_work_division(const BalanceLoopArgs<N>& args) {
    JasUnpack(args, loop, min_blocks, max_blocks);

    i32 leading_dim = loop.bounds[0];
    i64 trailing_dims = loop.num_iter / leading_dim;
    constexpr i32 min_work_per_block = 128;
    // NOTE(cmo): Launch a sensible number of blocks, but they need enough work
    while (leading_dim < min_blocks && trailing_dims >= min_work_per_block) {
        leading_dim *= 2;
        trailing_dims = (trailing_dims + 1) / 2; // ceiling div
    }
    // NOTE(cmo): Don't launch too many blocks
    while (leading_dim >= max_blocks) {
        leading_dim = (leading_dim + 1) / 2; // ceiling div
        trailing_dims *= 2;
    }
    // NOTE(cmo): however many blocks we launch, ensure there's enough work
    while (trailing_dims < min_work_per_block) {
        leading_dim = (leading_dim + 1) / 2; // ceiling div
        trailing_dims *= 2;
    }
    return TeamWorkDivision {
        .team_count = leading_dim,
        .inner_work_count = trailing_dims
    };
}

template <int N, class Lambda>
KOKKOS_INLINE_FUNCTION void array_invoke(const Lambda& closure, const Kokkos::Array<i32, N>& arr);


template <int N, class Lambda, typename T>
KOKKOS_INLINE_FUNCTION void array_invoke_with_ref_arg(
    const Lambda& closure,
    const Kokkos::Array<i32, N>& arr,
    T& ref
);

namespace DexImpl {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    constexpr int CudaVecLen = 128;
    template <int N, class Lambda>
    __global__ __launch_bounds__(CudaVecLen)
    void flat_loop_kernel(const FlatLoop<N> loop, const Lambda closure) {
        i64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < loop.num_iter) {
            auto idxs = loop.unpack(idx);
            array_invoke(closure, idxs);
        }
    }

    template <int N, class Lambda>
    inline void parallel_for_cuda_like(const FlatLoop<N>& loop, const Lambda& closure) {
        u32 block_dim = CudaVecLen;
        u64 grid_dim = (loop.num_iter + block_dim - 1) / block_dim;
#ifdef DEXRT_DEBUG
        KOKKOS_ASSERT(grid_dim < UINT32_MAX && "Grid dim too big!");
#endif
        flat_loop_kernel <<< u32(grid_dim), block_dim, 0, 0 >>> (loop, closure);
    }
#endif
}

template <class ExecutionSpace=Kokkos::DefaultExecutionSpace, bool DexFlatLoop=true, int N, class Lambda>
inline void dex_parallel_for(const std::string& name, const FlatLoop<N>& loop, const Lambda& closure) {
    static_assert(N < 7, "Flat loops only supported for 1 <= N <= 6");
#ifdef YAKL_AUTO_PROFILE
    yakl::timer_start(name);
#endif
#ifdef DEX_BALANCE_TEAM_WORK
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
#else
    #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if constexpr (DexFlatLoop && std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
        DexImpl::parallel_for_cuda_like(loop, closure);
    } else {
        Kokkos::parallel_for(
            name,
            Kokkos::RangePolicy<ExecutionSpace>(0, loop.num_iter),
            KOKKOS_LAMBDA (i64 idx) {
                auto idxs = loop.unpack(idx);
                array_invoke(closure, idxs);
            }
        );
    }
    #else
    Kokkos::parallel_for(
        name,
        loop.num_iter,
        KOKKOS_LAMBDA (i64 idx) {
            auto idxs = loop.unpack(idx);
            array_invoke(closure, idxs);
        }
    );
    #endif
#endif
#ifdef YAKL_AUTO_FENCE
    Kokkos::fence();
#endif
#ifdef YAKL_AUTO_PROFILE
    yakl::timer_stop(name);
#endif
}

template <class ExecutionSpace=Kokkos::DefaultExecutionSpace, bool DexFlatLoop=true, int N, class Lambda>
inline void dex_parallel_for(const FlatLoop<N>& loop, const Lambda& closure) {
    dex_parallel_for<ExecutionSpace, DexFlatLoop>("Unnamed Kernel", loop, closure);
}

namespace DexImpl {
    // NOTE(cmo): I just wanted to get a Kokkos::reducer acting on the same
    // types in a different execution space. If they add ones with more than two
    // args, we need a new override here.  Here be dragons.
    template <int N, class ExecutionSpace, template <typename...> typename Reducer, typename... RedArgs>
    struct reducer_type_impl;

    template <class ExecutionSpace, template <typename, typename> typename Reducer, typename RedArg0, typename RedSpace>
    struct reducer_type_impl<2, ExecutionSpace, Reducer, RedArg0, RedSpace> {
        typedef Reducer<RedArg0, ExecutionSpace> type;
    };

    template <class ExecutionSpace, template <typename, typename, typename> typename Reducer, typename RedArg0, typename RedArg1, typename RedSpace>
    struct reducer_type_impl<3, ExecutionSpace, Reducer, RedArg0, RedArg1, RedSpace> {
        typedef Reducer<RedArg0, RedArg1, ExecutionSpace> type;
    };

    template <class ExecutionSpace, template <typename...> class Reducer, typename... RedArgs>
    struct reducer_type_in_space {
        typedef typename reducer_type_impl<sizeof...(RedArgs), ExecutionSpace, Reducer, RedArgs...>::type type;
    };
};

template <
    class ExecutionSpace=Kokkos::DefaultExecutionSpace,
    int N,
    class Lambda,
    template<typename...> class Reducer,
    typename... Args
>
inline void dex_parallel_reduce(
    const std::string& name,
    const FlatLoop<N>& loop,
    const Lambda& closure,
    const Reducer<Args...>& reducer
) {
#ifdef DEX_BALACE_TEAM_WORK
    typedef Reducer<Args...> ReducerT;
    typedef typename ReducerT::value_type ReductionVar;
    typedef typename DexImpl::reducer_type_in_space<ExecutionSpace, Reducer, Args...>::type ReducerTDev;

    const auto work_div = balance_parallel_work_division(BalanceLoopArgs{.loop=loop});
    ReductionVar rvar;
    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<ExecutionSpace>(work_div.team_count, Kokkos::AUTO()),
        KOKKOS_LAMBDA (const Kokkos::TeamPolicy<ExecutionSpace>::member_type& team, ReductionVar& team_rvar) {
            const i64 i_base = team.league_rank() * work_div.inner_work_count;
            const i64 i_max = std::min(i_base + work_div.inner_work_count, loop.num_iter);
            const i32 inner_iter_count = i_max - i_base;
            if (inner_iter_count <= 0) {
                return;
            }

            ReductionVar thread_rvar;
            ReducerTDev thread_reducer(thread_rvar);

            Kokkos::parallel_reduce(
                Kokkos::TeamVectorRange(team, inner_iter_count),
                [&] (const int inner_i, ReductionVar& inner_rvar) {
                    const auto idxs = loop.unpack(i_base + inner_i);
                    array_invoke_with_ref_arg(closure, idxs, inner_rvar);
                },
                thread_reducer
            );

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                thread_reducer.join(team_rvar, thread_rvar);
            });
        },
        reducer
    );
#else
    typedef Reducer<Args...> ReducerT;
    typedef typename ReducerT::value_type ReductionVar;

    ReductionVar rvar;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>(0, loop.num_iter),
        KOKKOS_LAMBDA (const i64 idx, ReductionVar& rvar) {
            const auto idxs = loop.unpack(idx);
            array_invoke_with_ref_arg(closure, idxs, rvar);
        },
        reducer
    );
#endif
}

template <
    class ExecutionSpace=Kokkos::DefaultExecutionSpace,
    int N,
    class Lambda,
    template<typename...> class Reducer,
    typename... Args
>
inline void dex_parallel_reduce(
    const FlatLoop<N>& loop,
    const Lambda& closure,
    const Reducer<Args...>& reducer
) {
    dex_parallel_reduce("Unnamed reduction", loop, closure, reducer);
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

namespace Kokkos {
    template<>
    struct reduction_identity<Kokkos::pair<i32, i32>> {
        KOKKOS_FORCEINLINE_FUNCTION static Kokkos::pair<i32, i32> min() {
            return Kokkos::make_pair(0, 0);
        }
    };
}

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

template <class Lambda, typename T>
KOKKOS_INLINE_FUNCTION void array_invoke_with_ref_arg(
    const Lambda& closure,
    const Kokkos::Array<i32, 1>& arr,
    T& ref
) {
    closure(arr[0], ref);
}

template <class Lambda, typename T>
KOKKOS_INLINE_FUNCTION void array_invoke_with_ref_arg(
    const Lambda& closure,
    const Kokkos::Array<i32, 2>& arr,
    T& ref
) {
    closure(arr[0], arr[1], ref);
}

template <class Lambda, typename T>
KOKKOS_INLINE_FUNCTION void array_invoke_with_ref_arg(
    const Lambda& closure,
    const Kokkos::Array<i32, 3>& arr,
    T& ref
) {
    closure(arr[0], arr[1], arr[2], ref);
}

template <class Lambda, typename T>
KOKKOS_INLINE_FUNCTION void array_invoke_with_ref_arg(
    const Lambda& closure,
    const Kokkos::Array<i32, 4>& arr,
    T& ref
) {
    closure(arr[0], arr[1], arr[2], arr[3]);
}

template <class Lambda, typename T>
KOKKOS_INLINE_FUNCTION void array_invoke_with_ref_arg(
    const Lambda& closure,
    const Kokkos::Array<i32, 5>& arr,
    T& ref
) {
    closure(arr[0], arr[1], arr[2], arr[3], arr[4], ref);
}

template <class Lambda, typename T>
KOKKOS_INLINE_FUNCTION void array_invoke_with_ref_arg(
    const Lambda& closure,
    const Kokkos::Array<i32, 6>& arr,
    T& ref
) {
    closure(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], ref);
}
#else
#endif