#if !defined(DEXRT_RC_CONSTANTS_HPP)
#define DEXRT_RC_CONSTANTS_HPP
#include "Types.hpp"

constexpr int RC_DYNAMIC = (1 << 0);
constexpr int RC_PREAVERAGE = (1 << 1);
constexpr int RC_SAMPLE_BC = (1 << 2); // NOTE(cmo): Only affects raymarching
constexpr int RC_COMPUTE_ALO = (1 << 3); // NOTE(cmo): Only affects raymarching
constexpr int RC_DIR_BY_DIR = (1 << 4);
constexpr int RC_LINE_SWEEP = (1 << 5); // NOTE(cmo) only added in one place to flag for BC handling
constexpr int RC_PERIODIC = (1 << 6); // NOTE(cmo): Only affects raymarching

struct RcFlags {
    bool dynamic = false;
    bool preaverage = PREAVERAGE;
    bool sample_bc = false;
    bool compute_alo = false;
    bool dir_by_dir = DIR_BY_DIR;
    bool periodic = false;
} ;


YAKL_INLINE constexpr int RC_flags_pack(const RcFlags& flags) {
    int flag = 0;
    if (flags.dynamic) {
        flag |= RC_DYNAMIC;
    }
    if (flags.preaverage) {
        flag |= RC_PREAVERAGE;
    }
    if (flags.sample_bc) {
        flag |= RC_SAMPLE_BC;
    }
    if (flags.compute_alo) {
        flag |= RC_COMPUTE_ALO;
    }
    if (flags.dir_by_dir) {
        flag |= RC_DIR_BY_DIR;
    }
    if (flags.periodic) {
        flag |= RC_PERIODIC;
    }
    return flag;
}

#else
#endif