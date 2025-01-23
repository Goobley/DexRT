#if !defined(DEXRT_LOOP_UTILS_HPP)
#define DEXRT_LOOP_UTILS_HPP

#include "Config.hpp"
#include "Types.hpp"

/// All of this is C-style Iterate::Right loops.

template <int N>
struct LoopBoundsArgs {
    int bounds[N];
    int min_blocks = 256; /// Minimum number of blocks to use, reshuffle from higher indices as needed.
    int max_blocks = 65535; /// Maximum number of blocks, shuffle to lower indices as needed.
};

template <int N>
struct FlatLoop {
    Kokkos::Array<i32, N> bounds;

};

#else
#endif