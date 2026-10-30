#include "ScHalo.hpp"
#include "LoopUtils.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr i32 H = SC_MEDIUM_HALO;

/// Copies rows rows(i) of f to/from buf ([lead, n, row_size]).
template <typename T>
void copy_rows(const ScHaloField& f, const yakl::Array<i32, 1, yakl::memDevice>& rows, u8* buf, bool unpack) {
    T* data = static_cast<T*>(f.data);
    T* b = reinterpret_cast<T*>(buf);
    const i64 ls = f.lead_stride;
    const i64 rs = f.row_size;
    const i32 n = rows.extent(0);
    dex_parallel_for(
        "SC halo rows",
        FlatLoop<2>(i32(f.lead * n), i32(rs)),
        YAKL_LAMBDA (i32 lr, i32 j) {
            const i64 l = lr / n;
            const i32 r = lr - l * n;
            const i64 field = l * ls + (i64(rows(r)) + H) * rs + j;
            const i64 packed = i64(lr) * rs + j;
            if (unpack) {
                data[field] = b[packed];
            } else {
                b[packed] = data[field];
            }
        });
}

/// Copies rows src(i) of f to rows dst(i).
template <typename T>
void copy_local(const ScHaloField& f, const yakl::Array<i32, 1, yakl::memDevice>& src, const yakl::Array<i32, 1, yakl::memDevice>& dst) {
    T* data = static_cast<T*>(f.data);
    const i64 ls = f.lead_stride;
    const i64 rs = f.row_size;
    const i32 n = src.extent(0);
    dex_parallel_for(
        "SC halo rows (local)",
        FlatLoop<2>(i32(f.lead * n), i32(rs)),
        YAKL_LAMBDA (i32 lr, i32 j) {
            const i64 l = lr / n;
            const i32 r = lr - l * n;
            data[l * ls + (i64(dst(r)) + H) * rs + j] = data[l * ls + (i64(src(r)) + H) * rs + j];
        });
}

template <template <typename> class Op, typename... Args>
void by_size(const ScHaloField& f, Args&&... args) {
    switch (f.bytes) {
        case 1: Op<u8>{}(f, args...); break;
        case 2: Op<u16>{}(f, args...); break;
        case 4: Op<u32>{}(f, args...); break;
        case 8: Op<u64>{}(f, args...); break;
        default: throw std::runtime_error("SC halo fields have 1, 2, 4 or 8 byte elements");
    }
}
template <typename T>
struct CopyRows {
    template <typename... Args>
    void operator()(const ScHaloField& f, Args&&... args) {
        copy_rows<T>(f, args...);
    }
};
template <typename T>
struct CopyLocal {
    template <typename... Args>
    void operator()(const ScHaloField& f, Args&&... args) {
        copy_local<T>(f, args...);
    }
};

yakl::Array<i32, 1, yakl::memDevice> device_rows(const std::vector<i32>& rows, const char* label) {
    yakl::Array<i32, 1, yakl::memHost> h(label, rows.size());
    std::copy(rows.begin(), rows.end(), h.begin());
    return h.createDeviceCopy();
}

/// Field offsets in a buffer of n rows (8-byte aligned), and its size.
std::vector<i64> offsets(const std::vector<ScHaloField>& fields, i32 n) {
    std::vector<i64> result;
    i64 off = 0;
    for (const auto& f : fields) {
        result.push_back(off);
        off += (f.lead * n * f.row_size * f.bytes + 7) / 8 * 8;
    }
    result.push_back(off);
    return result;
}

template <typename T>
ScHaloField ks_field(T* data, i64 lead, const ScCells& cells) {
    return ScHaloField{.data = data, .lead = lead, .lead_stride = cells.num_cells(), .row_size = cells.plane_size(), .bytes = i32(sizeof(T))};
}

} // namespace

ScHalo::~ScHalo() {
#ifdef HAVE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comm);
    }
#endif
}

void ScHalo::init(const ScState& state) {
    const ScSlab& slab = state.slab;
    peers.clear();
    self_src = self_dst = yakl::Array<i32, 1, yakl::memDevice>();
    self_src_host.clear();
    self_dst_host.clear();
    if (slab.num_ranks == 1) {
        return;
    }
#ifdef HAVE_MPI
    if (comm == MPI_COMM_NULL) {
        MPI_Comm_dup(state.mpi_state.comm, &comm);
    }
    std::vector<i32> bounds(2 * slab.num_ranks);
    i32 local[2] = {slab.z_start, slab.z_end};
    MPI_Allgather(local, 2, MPI_INT, bounds.data(), 2, MPI_INT, comm);
    const i32 nz = slab.global_num_z;
    auto owner = [&](i32 z) {
        for (i32 r = 0; r < slab.num_ranks; ++r) {
            if (z >= bounds[2 * r] && z < bounds[2 * r + 1]) {
                return r;
            }
        }
        throw std::runtime_error("SC halo: slabs don't cover the domain");
    };
    // NOTE(cmo): Rank r's halo rows (global), each with its source row; every
    // rank enumerates them in the same order, which is the message order.
    auto halo_rows = [&](i32 r, auto&& visit) {
        const i32 s = bounds[2 * r], e = bounds[2 * r + 1];
        for (i32 g = s - H; g < e + H; ++g) {
            if (g == s) {
                g = e;
            }
            visit(g - s, std::clamp(g, 0, nz - 1));
        }
    };
    std::vector<std::vector<i32>> send(slab.num_ranks), recv(slab.num_ranks);
    std::vector<i32> self_s, self_d;
    for (i32 r = 0; r < slab.num_ranks; ++r) {
        halo_rows(r, [&](i32 dst, i32 src) {
            const i32 o = owner(src);
            if (r == slab.rank && o == slab.rank) {
                self_s.push_back(src - slab.z_start);
                self_d.push_back(dst);
            } else if (r == slab.rank) {
                recv[o].push_back(dst);
            } else if (o == slab.rank) {
                send[r].push_back(src - slab.z_start);
            }
        });
    }
    for (i32 r = 0; r < slab.num_ranks; ++r) {
        if (!send[r].empty() || !recv[r].empty()) {
            peers.push_back(Peer{.rank = r,
                .send_rows = device_rows(send[r], "sc_halo_send"),
                .recv_rows = device_rows(recv[r], "sc_halo_recv"),
                .send_host = send[r],
                .recv_host = recv[r]});
        }
    }
    if (!self_s.empty()) {
        self_src = device_rows(self_s, "sc_halo_self_src");
        self_dst = device_rows(self_d, "sc_halo_self_dst");
    }
    self_src_host = self_s;
    self_dst_host = self_d;
#endif
}

void ScHalo::exchange(const std::vector<ScHaloField>& fields) const {
    if ((peers.empty() && !self_src.initialized()) || fields.empty()) {
        return;
    }
#ifdef HAVE_MPI
    std::vector<yakl::Array<u8, 1, yakl::memHost>> sends(peers.size()), receives(peers.size());
    std::vector<MPI_Request> requests;
    for (size_t p = 0; p < peers.size(); ++p) {
        const Peer& peer = peers[p];
        const i32 recv_n = peer.recv_rows.extent(0);
        const i32 send_n = peer.send_rows.extent(0);
        const auto recv_off = offsets(fields, recv_n);
        const auto send_off = offsets(fields, send_n);
        if (std::max(recv_off.back(), send_off.back()) > std::numeric_limits<int>::max()) {
            throw std::runtime_error("SC halo message exceeds the MPI count limit");
        }
        if (recv_n) {
            receives[p] = yakl::Array<u8, 1, yakl::memHost>("sc_halo_recv", recv_off.back());
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Irecv(receives[p].data(), recv_off.back(), MPI_BYTE, peer.rank, 0, comm, &requests.back());
        }
        if (send_n) {
            yakl::Array<u8, 1, yakl::memDevice> packed("sc_halo_pack", send_off.back());
            for (size_t f = 0; f < fields.size(); ++f) {
                by_size<CopyRows>(fields[f], peer.send_rows, packed.data() + send_off[f], false);
            }
            Kokkos::fence();
            sends[p] = packed.createHostCopy();
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(sends[p].data(), send_off.back(), MPI_BYTE, peer.rank, 0, comm, &requests.back());
        }
    }
    if (self_src.initialized()) {
        for (const auto& f : fields) {
            by_size<CopyLocal>(f, self_src, self_dst);
        }
        Kokkos::fence();
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (size_t p = 0; p < peers.size(); ++p) {
        const Peer& peer = peers[p];
        const i32 recv_n = peer.recv_rows.extent(0);
        if (!recv_n) {
            continue;
        }
        const auto recv_off = offsets(fields, recv_n);
        const auto packed = receives[p].createDeviceCopy();
        for (size_t f = 0; f < fields.size(); ++f) {
            by_size<CopyRows>(fields[f], peer.recv_rows, packed.data() + recv_off[f], true);
        }
        Kokkos::fence();
    }
#endif
}

void ScHalo::exchange_static(ScState* state) const {
    if (state->slab.num_ranks == 1) {
        return;
    }
    auto& cells = state->cells;
    exchange({ScHaloField{.data = cells.active.data(), .lead = 1, .lead_stride = 0, .row_size = cells.plane_size(), .bytes = 1}});
    cells.build_lists();
    if (state->config.mode == DexrtMode::GivenFs) {
        auto& g = state->given_state;
        const i64 row = g.emis.extent(1) * g.emis.extent(2);
        exchange({
            ScHaloField{.data = g.emis.data(), .lead = 1, .lead_stride = 0, .row_size = row, .bytes = i32(sizeof(fp_t))},
            ScHaloField{.data = g.opac.data(), .lead = 1, .lead_stride = 0, .row_size = row, .bytes = i32(sizeof(fp_t))},
        });
    }
}

void ScHalo::exchange_inputs(ScState* state) const {
    if (state->slab.num_ranks > 1 && state->config.mode != DexrtMode::GivenFs) {
        const auto& cells = state->cells;
        auto& a = state->atmos;
        std::vector<ScHaloField> fields;
        for (Fp1d* f : {&a.temperature, &a.pressure, &a.ne, &a.nh_tot, &a.nh0, &a.vturb, &a.vx, &a.vy, &a.vz}) {
            fields.push_back(ks_field(f->data(), 1, cells));
        }
        fields.push_back(ks_field(state->pops.data(), state->pops.extent(0), cells));
        if (state->prd.enabled()) {
            fields.push_back(ks_field(state->prd.rho.data.data(), state->prd.rho.extent(0), cells));
        }
        auto& bg = state->background;
        if (bg.enabled) {
            fields.push_back(ks_field(bg.n1s.data(), 1, cells));
            fields.push_back(ks_field(bg.n_p.data(), 1, cells));
        }
        exchange(fields);
    }
    // Capture previous-J halos before this formal solution pages over J_cpu.
    // This also fills physical-edge halos on a single rank.
    if (state->background.enabled && state->background.have_J) {
        exchange_halo_rows_host(state->J_cpu, state->J_halo_cpu, state->cells.num_z, state->cells.plane_size());
    }
}

void ScHalo::exchange_halo_rows_host(const Rad16ArrayHost& owned, const Rad16ArrayHost& halo, i32 num_z, i64 plane) const {
    using T = Rad16ArrayHost::Storage;
    const i64 lead = owned.extent(0);
    const i64 os = owned.extent(1), hs = halo.extent(1);
    auto src = [&](i64 l, i32 z) {
        return owned.data.data() + l * os + i64(z) * plane;
    };
    auto dst = [&](i64 l, i32 z) {
        return halo.data.data() + l * hs + i64(z < 0 ? z + H : z - num_z + H) * plane;
    };
    for (i64 l = 0; l < lead; ++l) {
        for (size_t i = 0; i < self_src_host.size(); ++i) {
            std::copy_n(src(l, self_src_host[i]), plane, dst(l, self_dst_host[i]));
        }
    }
#ifdef HAVE_MPI
    std::vector<std::vector<T>> sends(peers.size()), receives(peers.size());
    std::vector<MPI_Request> requests;
    for (size_t p = 0; p < peers.size(); ++p) {
        const Peer& peer = peers[p];
        const i64 recv_count = lead * i64(peer.recv_host.size()) * plane * sizeof(T);
        const i64 send_count = lead * i64(peer.send_host.size()) * plane * sizeof(T);
        if (std::max(recv_count, send_count) > std::numeric_limits<int>::max()) {
            throw std::runtime_error("SC halo message exceeds the MPI count limit");
        }
        if (recv_count) {
            receives[p].resize(recv_count / sizeof(T));
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Irecv(receives[p].data(), recv_count, MPI_BYTE, peer.rank, 1, comm, &requests.back());
        }
        if (send_count) {
            auto& buf = sends[p];
            buf.resize(send_count / sizeof(T));
            T* out = buf.data();
            for (i64 l = 0; l < lead; ++l) {
                for (i32 z : peer.send_host) {
                    out = std::copy_n(src(l, z), plane, out);
                }
            }
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(buf.data(), send_count, MPI_BYTE, peer.rank, 1, comm, &requests.back());
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (size_t p = 0; p < peers.size(); ++p) {
        const T* in = receives[p].data();
        for (i64 l = 0; l < lead && !receives[p].empty(); ++l) {
            for (i32 z : peers[p].recv_host) {
                std::copy_n(in, plane, dst(l, z));
                in += plane;
            }
        }
    }
#endif
}
