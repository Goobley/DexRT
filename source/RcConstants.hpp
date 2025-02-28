#if !defined(DEXRT_RC_CONSTANTS_HPP)
#define DEXRT_RC_CONSTANTS_HPP
#include "Types.hpp"

constexpr int RC_DYNAMIC = 0x1;
constexpr int RC_PREAVERAGE = 0x2;
constexpr int RC_SAMPLE_BC = 0x4;
constexpr int RC_COMPUTE_ALO = 0x8;
constexpr int RC_DIR_BY_DIR = 0x10;
constexpr int RC_LINE_SWEEP = 0x20; // NOTE(cmo) only added in one place to flag for BC handling

struct RcFlags {
    bool dynamic = false;
    bool preaverage = PREAVERAGE;
    bool sample_bc = false;
    bool compute_alo = false;
    bool dir_by_dir = DIR_BY_DIR;
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
    return flag;
}

#else
#endif