// ScIo round trip on any number of ranks: writes dense, compact (with an empty
// rank) and owned blocks in tiny chunks, reads them back as different blocks,
// and leaves the file for comparison across backends and rank counts.
// Usage: mpirun -n R dexrt_sc_io_test OUT.nc Gather|Parallel [MB]
// With MB: instead write and read back an MB block per rank (default chunks),
// reporting each rank's peak RSS growth (rank 0's should not grow with MB).
#include "ScIo.hpp"
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace {

constexpr i64 NL = 3, NZ = 12, NX = 5;
constexpr i64 CHUNK = 24; /// bytes: a few values per message

f32 dense_value(i64 l, i64 z, i64 x) {
    return f32(l * 1000 + z * 10 + x) + 0.25f;
}

/// Rank r's compact cells: rank 1 has none, the others r + 2.
i64 compact_count(int r) {
    return r == 1 ? 0 : r + 2;
}

void check(bool cond, const char* what) {
    if (!cond) {
        throw std::runtime_error(what);
    }
}

/// Peak resident set size [MB].
f64 peak_rss_mb() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmHWM:") {
            f64 kb;
            status >> kb;
            return kb / 1024.0;
        }
    }
    return -1.0;
}

/// Write then read back an mb-sized dense block per rank.
int size_test(const MpiState& mpi, const std::string& path, ScIoBackend backend, i64 mb) {
    const i64 nx = 1024, rows = mb * (1 << 20) / (nx * 4);
    std::vector<f32> block(rows * nx, f32(mpi.rank + 1));
    const f64 base = peak_rss_mb();
    {
        ScOutFile file(mpi, path, backend);
        file.write_block("big", {"z", "x"}, {rows * mpi.num_ranks, nx}, 0, rows * mpi.rank, rows, block.data());
    }
    const f64 after_write = peak_rss_mb();
    {
        ScInFile in(mpi, path);
        in.read_block("big", 0, rows * mpi.rank, rows, block.data());
    }
    const f64 after_read = peak_rss_mb();
    int ok = block.front() == f32(mpi.rank + 1) && block.back() == f32(mpi.rank + 1);
    std::vector<f64> growth = {after_write - base, after_read - base};
    std::vector<f64> all(2 * mpi.num_ranks);
    MPI_Gather(growth.data(), 2, MPI_DOUBLE, all.data(), 2, MPI_DOUBLE, 0, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, mpi.comm);
    if (mpi.rank == 0) {
        for (int r = 0; r < mpi.num_ranks; ++r) {
            std::printf("%lld MB/rank: rank %d peak RSS growth: write %.0f MB, read %.0f MB\n", (long long)mb, r, all[2 * r], all[2 * r + 1]);
        }
    }
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MpiState mpi;
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi.comm);
    MPI_Comm_rank(mpi.comm, &mpi.rank);
    MPI_Comm_size(mpi.comm, &mpi.num_ranks);
    const std::string path = argv[1];
    const ScIoBackend backend = std::string(argv[2]) == "Parallel" ? ScIoBackend::Parallel : ScIoBackend::Gather;
    const int R = mpi.num_ranks;
    if (argc > 3) {
        const int status = size_test(mpi, path, backend, std::stoll(argv[3]));
        MPI_Finalize();
        return status;
    }
    int status = 0;
    try {
        // NOTE(cmo): Uneven z rows per rank (the last rank takes the rest).
        const i64 z0 = mpi.rank * (NZ / R), z1 = mpi.rank == R - 1 ? NZ : z0 + NZ / R;
        std::vector<f32> dense(NL * (z1 - z0) * NX);
        for (i64 l = 0; l < NL; ++l) {
            for (i64 z = z0; z < z1; ++z) {
                for (i64 x = 0; x < NX; ++x) {
                    dense[(l * (z1 - z0) + (z - z0)) * NX + x] = dense_value(l, z, x);
                }
            }
        }
        i64 offset = 0, total = 0;
        for (int r = 0; r < R; ++r) {
            total += compact_count(r);
            if (r < mpi.rank) {
                offset += compact_count(r);
            }
        }
        const i64 n = compact_count(mpi.rank);
        std::vector<f64> compact(NL * n);
        for (i64 l = 0; l < NL; ++l) {
            for (i64 k = 0; k < n; ++k) {
                compact[l * n + k] = f64(l) * 1e6 + f64(offset + k) * 1.5;
            }
        }
        std::vector<i32> owned = {7, 8, 9, 10};
        {
            ScOutFile file(mpi, path, backend, CHUNK);
            file.put_att("num_iter", i32(42));
            file.put_att("label", std::string(mpi.rank == 0 ? "rank 0 wins" : "ignored"));
            file.put_att("values", std::vector<f64>{1.0, 2.5});
            file.write_block("dense", {"level", "z", "x"}, {NL, NZ, NX}, 1, z0, z1 - z0, dense.data());
            file.write_block("compact", {"level", "ks"}, {NL, total}, 1, offset, n, compact.data());
            file.write_whole("owned", {"four"}, {4}, owned.data(), R - 1);
        }

        ScInFile in(mpi, path, CHUNK);
        int iter = 0;
        check(in.att_int("num_iter", &iter) && iter == 42, "attribute");
        check(in.shape("dense") == std::vector<i64>{NL, NZ, NX}, "shape");
        check(!in.has_var("missing") && in.dim("ks") == total, "metadata");
        check(in.read_whole<i32>("owned") == owned, "owned");
        // NOTE(cmo): Read back a different block from the one written: rows
        // reversed between ranks.
        const int q = R - 1 - mpi.rank;
        const i64 r0 = q * (NZ / R), r1 = q == R - 1 ? NZ : r0 + NZ / R;
        std::vector<f32> back(NL * (r1 - r0) * NX);
        in.read_block("dense", 1, r0, r1 - r0, back.data());
        for (i64 l = 0; l < NL; ++l) {
            for (i64 z = r0; z < r1; ++z) {
                for (i64 x = 0; x < NX; ++x) {
                    check(back[(l * (r1 - r0) + (z - r0)) * NX + x] == dense_value(l, z, x), "dense block");
                }
            }
        }
        const auto all = in.read_whole<f64>("compact");
        for (i64 l = 0; l < NL; ++l) {
            for (i64 k = 0; k < total; ++k) {
                check(all[l * total + k] == f64(l) * 1e6 + f64(k) * 1.5, "compact");
            }
        }
        std::vector<f32> plane(NL * NX);
        in.read_block("dense", 1, NZ - 1, 1, plane.data());
        check(plane[NX + 2] == dense_value(1, NZ - 1, 2), "top row");
    } catch (const std::exception& e) {
        std::printf("rank %d: FAIL %s\n", mpi.rank, e.what());
        status = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, mpi.comm);
    if (mpi.rank == 0) {
        std::printf("%s ScIo round trip on %d ranks (%s)\n", status ? "FAIL" : "PASS", R, argv[2]);
    }
    MPI_Finalize();
    return status;
}
