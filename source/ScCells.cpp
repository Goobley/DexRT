#include "ScCells.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <type_traits>

ScActiveMaskHost sc_active_mask_from_tiles(const ScActiveMaskHost& global_tiles,
    i32 z_start,
    i32 num_z,
    i32 tile,
    i32 ny) {
    const i64 global_z = i64(global_tiles.extent(0)) * tile;
    const i32 num_y_tiles = ny < 1 ? 0 : (ny + tile - 1) / tile;
    if (z_start < 0 || num_z < 1 || i64(z_start) + num_z > global_z || global_tiles.extent(1) < 1 ||
        num_y_tiles < 1 || global_tiles.extent(1) % num_y_tiles) {
        throw std::runtime_error("SC active tile window is outside the global grid");
    }
    const i32 nxt = global_tiles.extent(1) / num_y_tiles;
    const i32 nx = nxt * tile;
    ScActiveMaskHost mask("sc_active_window", num_z, i64(ny) * nx);
    for (i32 z = 0; z < num_z; ++z) {
        for (i32 y = 0; y < ny; ++y) {
            for (i32 x = 0; x < nx; ++x) {
                const u8 flag = global_tiles((z_start + z) / tile, (y / tile) * nxt + x / tile);
                if (flag > 1) {
                    throw std::runtime_error("SC tile activity must be 0 or 1");
                }
                mask(z, i64(y) * nx + x) = flag;
            }
        }
    }
    return mask;
}

void ScCells::init(const ScActiveMaskHost& mask, i32 global_z_start, i32 group, i32 ny) {
    const i32 nz = mask.extent(0);
    const i64 plane = mask.extent(1);
    if (nz < 1 || plane < 1 || ny < 1 || plane % ny || global_z_start < 0 || group < 1) {
        throw std::runtime_error("SC cells need positive extents/group and nonnegative origin");
    }
    num_x = plane / ny;
    num_y = ny;
    num_z = nz;
    z_start = global_z_start;
    group_size = group;
    yakl::Array<u8, 2, yakl::memHost> act("sc_active", num_rows(), plane);
    act = u8(0);
    for (i32 z = 0; z < nz; ++z) {
        for (i64 xy = 0; xy < plane; ++xy) {
            if (mask(z, xy) > 1) {
                throw std::runtime_error("SC cell activity must be 0 or 1");
            }
            act(z + SC_MEDIUM_HALO, xy) = mask(z, xy);
        }
    }
    active = act.createDeviceCopy();
    build_lists();
}

/// Split contiguous active spans into bounded pieces (owned rows first, then
/// the halo rows). Kernels must honour each piece's runtime length, including
/// short tails, rather than reading neighbouring inactive cells.
void ScCells::build_lists() {
    const auto act = active.createHostCopy();
    std::vector<i64> list, list_halo;
    std::vector<ScRunPiece> pieces, halo_pieces;
    for (i32 z = -SC_MEDIUM_HALO; z < num_z + SC_MEDIUM_HALO; ++z) {
        const bool own = z >= 0 && z < num_z;
        auto& p = own ? pieces : halo_pieces;
        for (i32 y = 0; y < num_y; ++y) {
            const i64 row = i64(y) * num_x;
            i32 x = 0;
            while (x < num_x) {
                if (!act(z + SC_MEDIUM_HALO, row + x)) {
                    ++x;
                    continue;
                }
                const i32 x0 = x;
                // NOTE(cmo): Pieces end at global multiples of group, so each lies
                // within one aligned group (a cache line on CPU).
                do {
                    if (own) {
                        list.push_back(ks(x, y, z));
                    }
                    list_halo.push_back(ks(x, y, z));
                    ++x;
                } while (x < num_x && act(z + SC_MEDIUM_HALO, row + x) && x % group_size != 0);
                p.push_back(ScRunPiece{.z = z, .y = y, .x0 = x0, .n = x - x0});
            }
        }
    }
    num_active = list.size();
    num_active_with_halo = list_halo.size();
    num_runs = pieces.size();
    pieces.insert(pieces.end(), halo_pieces.begin(), halo_pieces.end());
    num_runs_with_halo = pieces.size();
    auto device = [](const auto& v, const char* label) {
        yakl::Array<std::decay_t<decltype(v[0])>, 1, yakl::memHost> h(label, v.size());
        std::copy(v.begin(), v.end(), h.begin());
        return h.createDeviceCopy();
    };
    owned = device(list, "sc_owned");
    owned_and_halo = device(list_halo, "sc_owned_and_halo");
    runs = device(pieces, "sc_runs");
}
