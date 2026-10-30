#if !defined(DEXRT_SC_IO_HPP)
#define DEXRT_SC_IO_HPP

#include "Types.hpp"
#include "MpiWrapper.hpp"
#include "DexrtConfig.hpp"
#include <netcdf.h>
#include <string>
#include <vector>

/// File I/O for dexrt_sc. Every rank makes the same calls in the same order.
/// Data is exchanged as blocks: each rank holds rows [start, start + count)
/// along one axis of a variable (row-major, with count in place of that
/// axis's global extent). The z-slab decomposition makes each rank's rows one
/// contiguous hyperslab.


/// Upper bound on the extra memory rank 0 uses per buffer when streaming.
constexpr i64 SC_IO_CHUNK_BYTES = i64(256) << 20;

/// Whether this build supports ScIoBackend::Parallel.
bool sc_io_parallel_available();

class ScOutFile {
    MpiState mpi;
    bool parallel = false;
    bool open = false;
    int ncid = -1;
    i64 chunk_bytes;
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_NULL;
#endif

    int def_var(const std::string& name, nc_type type, const std::vector<std::string>& dims, const std::vector<i64>& shape);
    void put_att_raw(const std::string& name, nc_type type, i64 n, const void* data);

public:
    ScOutFile(const MpiState& mpi, const std::string& path, ScIoBackend backend, i64 chunk_bytes = SC_IO_CHUNK_BYTES);
    ~ScOutFile();
    ScOutFile(const ScOutFile&) = delete;
    ScOutFile& operator=(const ScOutFile&) = delete;
    void close();
    bool is_parallel() const { return parallel; }

    /// Global attributes take rank 0's value.
    void put_att(const std::string& name, const std::string& value);
    template <typename T>
    void put_att(const std::string& name, const std::vector<T>& value);
    template <typename T>
    void put_att(const std::string& name, T value) {
        put_att(name, std::vector<T>{value});
    }

    /// Write this rank's block [start, start + count) along axis split of the
    /// variable name[dims] with global extents shape.
    template <typename T>
    void write_block(
        const std::string& name,
        const std::vector<std::string>& dims,
        const std::vector<i64>& shape,
        int split,
        i64 start,
        i64 count,
        const T* data
    );

    /// Write a whole variable held by owner (other ranks' data unused).
    template <typename T>
    void write_whole(
        const std::string& name,
        const std::vector<std::string>& dims,
        const std::vector<i64>& shape,
        const T* data,
        int owner = 0
    ) {
        const bool mine = mpi.rank == owner;
        write_block(name, dims, shape, 0, 0, mine ? shape[0] : 0, data);
    }

    /// A host array held by rank 0 (as yakl::SimpleNetCDF::write).
    template <typename Array>
    void write(const Array& a, const std::string& name, const std::vector<std::string>& dims) {
        std::vector<i64> shape(a.get_rank());
        for (int i = 0; i < a.get_rank(); ++i) {
            shape[i] = a.extent(i);
        }
        write_whole(name, dims, shape, a.data(), 0);
    }
};

class ScInFile {
    MpiState mpi;
    int ncid = -1;
    i64 chunk_bytes;
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_NULL;
#endif

    /// varid and shape on rank 0, shape broadcast (empty if absent).
    std::vector<i64> shape_of(const std::string& name, int* varid);

public:
    /// Only rank 0 opens path.
    ScInFile(const MpiState& mpi, const std::string& path, i64 chunk_bytes = SC_IO_CHUNK_BYTES);
    ~ScInFile();
    ScInFile(const ScInFile&) = delete;
    ScInFile& operator=(const ScInFile&) = delete;

    bool has_var(const std::string& name);
    bool has_dim(const std::string& name);
    /// -1 if absent.
    i64 dim(const std::string& name);
    /// Global extents of name (empty if absent).
    std::vector<i64> shape(const std::string& name);
    /// Whether the global attribute exists; value set if so.
    bool att_int(const std::string& name, int* value);

    /// The whole variable, on every rank.
    template <typename T>
    std::vector<T> read_whole(const std::string& name);
    template <typename T>
    T read_scalar(const std::string& name) {
        return read_whole<T>(name).at(0);
    }
    /// This rank's block [start, start + count) along axis split of name.
    template <typename T>
    void read_block(const std::string& name, int split, i64 start, i64 count, T* out);
};

/// Opens path read-only on every rank. With several ranks, rank 0 reads the
/// file and broadcasts it, and the others open the copy in memory (buffer
/// must outlive ncid). For small replicated files (e.g. tables).
int sc_nc_open_replicated(const MpiState& mpi, const std::string& path, int* ncid, std::vector<char>* buffer);

#else
#endif
