#include "ScIo.hpp"
#include <netcdf_mem.h>
#if defined(HAVE_MPI) && defined(DEXRT_HAVE_NC_PAR)
    #include <netcdf_par.h>
#endif
#include <fmt/core.h>
#include <algorithm>
#include <climits>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace {

void nc_check(int err, const std::string& what) {
    if (err != NC_NOERR) {
        throw std::runtime_error(fmt::format("SC I/O ({}): {}", what, nc_strerror(err)));
    }
}

// NOTE(cmo): Overloaded on the fundamental types (as YAKL_netcdf.h), so the
// fixed-width aliases (e.g. i64 as long or long long) resolve on any platform.
nc_type nc_type_of(const float*) { return NC_FLOAT; }
nc_type nc_type_of(const double*) { return NC_DOUBLE; }
nc_type nc_type_of(const int*) { return NC_INT; }
nc_type nc_type_of(const long*) { static_assert(sizeof(long) == 8); return NC_INT64; }
nc_type nc_type_of(const long long*) { return NC_INT64; }
nc_type nc_type_of(const unsigned short*) { return NC_USHORT; }
nc_type nc_type_of(const unsigned char*) { return NC_UBYTE; }
template <typename T>
nc_type nc_type_of() {
    return nc_type_of(static_cast<const T*>(nullptr));
}

/// Typed (converting) reads.
int nc_get(int ncid, int var, const size_t* st, const size_t* ct, float* out) { return nc_get_vara_float(ncid, var, st, ct, out); }
int nc_get(int ncid, int var, const size_t* st, const size_t* ct, double* out) { return nc_get_vara_double(ncid, var, st, ct, out); }
int nc_get(int ncid, int var, const size_t* st, const size_t* ct, int* out) { return nc_get_vara_int(ncid, var, st, ct, out); }
int nc_get(int ncid, int var, const size_t* st, const size_t* ct, long* out) { return nc_get_vara_long(ncid, var, st, ct, out); }
int nc_get(int ncid, int var, const size_t* st, const size_t* ct, long long* out) { return nc_get_vara_longlong(ncid, var, st, ct, out); }
int nc_get(int ncid, int var, const size_t* st, const size_t* ct, unsigned short* out) { return nc_get_vara_ushort(ncid, var, st, ct, out); }
int nc_get(int ncid, int var, const size_t* st, const size_t* ct, unsigned char* out) { return nc_get_vara_uchar(ncid, var, st, ct, out); }

#ifdef HAVE_MPI
MPI_Datatype mpi_type_of(const float*) { return MPI_FLOAT; }
MPI_Datatype mpi_type_of(const double*) { return MPI_DOUBLE; }
MPI_Datatype mpi_type_of(const int*) { return MPI_INT; }
MPI_Datatype mpi_type_of(const long*) { return MPI_LONG; }
MPI_Datatype mpi_type_of(const long long*) { return MPI_LONG_LONG; }
MPI_Datatype mpi_type_of(const unsigned short*) { return MPI_UNSIGNED_SHORT; }
MPI_Datatype mpi_type_of(const unsigned char*) { return MPI_UNSIGNED_CHAR; }
MPI_Datatype mpi_type_of(const char*) { return MPI_CHAR; }
template <typename T>
MPI_Datatype mpi_type_of() {
    return mpi_type_of(static_cast<const T*>(nullptr));
}

/// Broadcast from rank 0, in pieces below the int count limit.
template <typename T>
void bcast(T* data, i64 n, MPI_Comm comm) {
    constexpr i64 piece = INT_MAX / 2;
    for (i64 i = 0; i < n; i += piece) {
        MPI_Bcast(data + i, int(std::min(piece, n - i)), mpi_type_of<T>(), 0, comm);
    }
}

template <typename T>
void bcast_vector(std::vector<T>* v, MPI_Comm comm) {
    i64 n = v->size();
    MPI_Bcast(&n, 1, MPI_INT64_T, 0, comm);
    v->resize(n);
    bcast(v->data(), n, comm);
}
#endif

/// A rank's block [start, start + count) along split, walked in chunks of
/// rows along axis 0 of at most chunk_bytes.
struct ScBlock {
    std::vector<i64> ext; /// the block's extents
    i64 start;
    int split;
    i64 row_elems; /// per index of axis 0
    i64 rows;
    i64 rows_per_chunk;

    ScBlock(const std::vector<i64>& shape, int split_, i64 start_, i64 count, i64 elem_bytes, i64 chunk_bytes)
        : ext(shape), start(start_), split(split_) {
        if (split < 0 || split >= int(shape.size()) || start < 0 || count < 0 || start + count > shape[split]) {
            throw std::runtime_error("SC I/O: block outside its variable");
        }
        ext[split] = count;
        row_elems = 1;
        for (int i = 1; i < ext.size(); ++i) {
            row_elems *= ext[i];
        }
        rows = (count == 0 || row_elems == 0) ? 0 : ext[0];
        rows_per_chunk = std::max<i64>(1, chunk_bytes / std::max<i64>(1, row_elems * elem_bytes));
        if (rows > 0 && std::min(rows, rows_per_chunk) * row_elems > INT_MAX) {
            throw std::runtime_error("SC I/O: a row is too large to stream");
        }
    }

    i64 num_chunks() const {
        return (rows + rows_per_chunk - 1) / rows_per_chunk;
    }
    i64 chunk_rows(i64 c) const {
        return std::min(rows_per_chunk, rows - c * rows_per_chunk);
    }
    /// The file hyperslab of chunk c.
    void hyperslab(i64 c, std::vector<size_t>* st, std::vector<size_t>* ct) const {
        st->assign(ext.size(), 0);
        ct->assign(ext.begin(), ext.end());
        (*st)[split] = start;
        (*st)[0] += c * rows_per_chunk;
        (*ct)[0] = chunk_rows(c);
    }
};

} // namespace

bool sc_io_parallel_available() {
#if defined(HAVE_MPI) && defined(DEXRT_HAVE_NC_PAR)
    return true;
#else
    return false;
#endif
}

ScOutFile::ScOutFile(const MpiState& mpi_, const std::string& path, ScIoBackend backend, i64 chunk_bytes_)
    : mpi(mpi_), chunk_bytes(chunk_bytes_) {
    if (backend == ScIoBackend::Parallel && !sc_io_parallel_available()) {
        throw std::runtime_error("SC I/O: this build has no parallel netCDF (output.io: Parallel)");
    }
    if (backend == ScIoBackend::Auto) {
        backend = sc_io_parallel_available() ? ScIoBackend::Parallel : ScIoBackend::Gather;
    }
    parallel = backend == ScIoBackend::Parallel && mpi.num_ranks > 1;
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        MPI_Comm_dup(mpi.comm, &comm);
    }
#endif
    if (parallel) {
#if defined(HAVE_MPI) && defined(DEXRT_HAVE_NC_PAR)
        nc_check(nc_create_par(path.c_str(), NC_NETCDF4 | NC_CLOBBER, comm, MPI_INFO_NULL, &ncid), path);
#endif
    } else if (mpi.rank == 0) {
        nc_check(nc_create(path.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid), path);
    }
    open = true;
}

ScOutFile::~ScOutFile() {
    if (open) {
        close();
    }
}

void ScOutFile::close() {
    if (!open) {
        return;
    }
    if (parallel || mpi.rank == 0) {
        nc_check(nc_close(ncid), "close");
    }
#ifdef HAVE_MPI
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comm);
    }
#endif
    open = false;
}

int ScOutFile::def_var(const std::string& name, nc_type type, const std::vector<std::string>& dims, const std::vector<i64>& shape) {
    std::vector<int> dimids(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
        if (nc_inq_dimid(ncid, dims[i].c_str(), &dimids[i]) == NC_NOERR) {
            size_t len;
            nc_check(nc_inq_dimlen(ncid, dimids[i], &len), dims[i]);
            if (len != size_t(shape[i])) {
                throw std::runtime_error(fmt::format("SC I/O: dimension {} is {}, not {} (for {})", dims[i], len, shape[i], name));
            }
        } else {
            nc_check(nc_def_dim(ncid, dims[i].c_str(), shape[i], &dimids[i]), dims[i]);
        }
    }
    int var;
    nc_check(nc_def_var(ncid, name.c_str(), type, dims.size(), dimids.data(), &var), name);
#if defined(HAVE_MPI) && defined(DEXRT_HAVE_NC_PAR)
    if (parallel) {
        nc_check(nc_var_par_access(ncid, var, NC_COLLECTIVE), name);
    }
#endif
    return var;
}

void ScOutFile::put_att_raw(const std::string& name, nc_type type, i64 n, const void* data) {
    if (parallel || mpi.rank == 0) {
        nc_check(nc_put_att(ncid, NC_GLOBAL, name.c_str(), type, n, data), name);
    }
}

void ScOutFile::put_att(const std::string& name, const std::string& value) {
    std::vector<char> v(value.begin(), value.end());
#ifdef HAVE_MPI
    if (parallel) {
        bcast_vector(&v, comm);
    }
#endif
    if (parallel || mpi.rank == 0) {
        nc_check(nc_put_att_text(ncid, NC_GLOBAL, name.c_str(), v.size(), v.data()), name);
    }
}

template <typename T>
void ScOutFile::put_att(const std::string& name, const std::vector<T>& value) {
    std::vector<T> v(value);
#ifdef HAVE_MPI
    if (parallel) {
        bcast_vector(&v, comm);
    }
#endif
    put_att_raw(name, nc_type_of<T>(), v.size(), v.data());
}

template <typename T>
void ScOutFile::write_block(
    const std::string& name,
    const std::vector<std::string>& dims,
    const std::vector<i64>& shape,
    int split,
    i64 start,
    i64 count,
    const T* data
) {
    if (dims.size() != shape.size() || shape.empty()) {
        throw std::runtime_error(fmt::format("SC I/O: {} needs one dimension name per extent", name));
    }
    int var = -1;
    if (parallel || mpi.rank == 0) {
        var = def_var(name, nc_type_of<T>(), dims, shape);
    }
    std::vector<size_t> st, ct;
    const ScBlock mine(shape, split, start, count, sizeof(T), chunk_bytes);
    if (mpi.num_ranks == 1) {
        for (i64 c = 0; c < mine.num_chunks(); ++c) {
            mine.hyperslab(c, &st, &ct);
            nc_check(nc_put_vara(ncid, var, st.data(), ct.data(), data + c * mine.rows_per_chunk * mine.row_elems), name);
        }
        return;
    }
#ifdef HAVE_MPI
    if (parallel) {
        // NOTE(cmo): Collective: every rank makes the same number of calls,
        // empty ones once its own block is done.
        i64 num_calls = mine.num_chunks();
        MPI_Allreduce(MPI_IN_PLACE, &num_calls, 1, MPI_INT64_T, MPI_MAX, comm);
        const std::vector<size_t> zero(shape.size(), 0);
        for (i64 c = 0; c < num_calls; ++c) {
            if (c < mine.num_chunks()) {
                mine.hyperslab(c, &st, &ct);
                nc_check(nc_put_vara(ncid, var, st.data(), ct.data(), data + c * mine.rows_per_chunk * mine.row_elems), name);
            } else {
                nc_check(nc_put_vara(ncid, var, zero.data(), zero.data(), data), name);
            }
        }
        return;
    }
    // NOTE(cmo): Gather: rank 0 writes the ranks' blocks in turn, receiving the
    // next chunk while it writes the current one.
    i64 mine_sc[2] = {start, count};
    std::vector<i64> all(2 * mpi.num_ranks);
    MPI_Gather(mine_sc, 2, MPI_INT64_T, all.data(), 2, MPI_INT64_T, 0, comm);
    const MPI_Datatype type = mpi_type_of<T>();
    if (mpi.rank != 0) {
        for (i64 c = 0; c < mine.num_chunks(); ++c) {
            MPI_Send(data + c * mine.rows_per_chunk * mine.row_elems, int(mine.chunk_rows(c) * mine.row_elems), type, 0, 0, comm);
        }
        return;
    }
    std::vector<T> buf[2];
    for (int r = 0; r < mpi.num_ranks; ++r) {
        const ScBlock b(shape, split, all[2 * r], all[2 * r + 1], sizeof(T), chunk_bytes);
        const i64 n = b.num_chunks();
        if (r == 0) {
            for (i64 c = 0; c < n; ++c) {
                b.hyperslab(c, &st, &ct);
                nc_check(nc_put_vara(ncid, var, st.data(), ct.data(), data + c * b.rows_per_chunk * b.row_elems), name);
            }
            continue;
        }
        MPI_Request req = MPI_REQUEST_NULL;
        auto post = [&](i64 c) {
            auto& dst = buf[c % 2];
            dst.resize(b.chunk_rows(c) * b.row_elems);
            MPI_Irecv(dst.data(), int(dst.size()), type, r, 0, comm, &req);
        };
        if (n > 0) {
            post(0);
        }
        for (i64 c = 0; c < n; ++c) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            if (c + 1 < n) {
                post(c + 1);
            }
            b.hyperslab(c, &st, &ct);
            nc_check(nc_put_vara(ncid, var, st.data(), ct.data(), buf[c % 2].data()), name);
        }
    }
#endif
}

ScInFile::ScInFile(const MpiState& mpi_, const std::string& path, i64 chunk_bytes_)
    : mpi(mpi_), chunk_bytes(chunk_bytes_) {
    int err = NC_NOERR;
    if (mpi.rank == 0) {
        err = nc_open(path.c_str(), NC_NOWRITE, &ncid);
    }
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        MPI_Comm_dup(mpi.comm, &comm);
        MPI_Bcast(&err, 1, MPI_INT, 0, comm);
    }
#endif
    nc_check(err, path);
}

ScInFile::~ScInFile() {
    if (mpi.rank == 0 && ncid >= 0) {
        nc_close(ncid);
    }
#ifdef HAVE_MPI
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comm);
    }
#endif
}

std::vector<i64> ScInFile::shape_of(const std::string& name, int* varid) {
    // NOTE(cmo): [-1] if absent; scalars are [] (as are 0-d variables).
    std::vector<i64> result;
    if (mpi.rank == 0) {
        if (nc_inq_varid(ncid, name.c_str(), varid) != NC_NOERR) {
            result = {-1};
        } else {
            int ndims;
            nc_check(nc_inq_varndims(ncid, *varid, &ndims), name);
            std::vector<int> dimids(ndims);
            nc_check(nc_inq_vardimid(ncid, *varid, dimids.data()), name);
            for (int d : dimids) {
                size_t len;
                nc_check(nc_inq_dimlen(ncid, d, &len), name);
                result.push_back(len);
            }
        }
    }
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        bcast_vector(&result, comm);
    }
#endif
    return result;
}

bool ScInFile::has_var(const std::string& name) {
    int varid;
    const auto s = shape_of(name, &varid);
    return !(s.size() == 1 && s[0] == -1);
}

std::vector<i64> ScInFile::shape(const std::string& name) {
    int varid;
    auto s = shape_of(name, &varid);
    if (s.size() == 1 && s[0] == -1) {
        return {};
    }
    return s;
}

i64 ScInFile::dim(const std::string& name) {
    i64 len = -1;
    if (mpi.rank == 0) {
        int id;
        size_t l;
        if (nc_inq_dimid(ncid, name.c_str(), &id) == NC_NOERR && nc_inq_dimlen(ncid, id, &l) == NC_NOERR) {
            len = l;
        }
    }
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        MPI_Bcast(&len, 1, MPI_INT64_T, 0, comm);
    }
#endif
    return len;
}

bool ScInFile::has_dim(const std::string& name) {
    return dim(name) >= 0;
}

bool ScInFile::att_int(const std::string& name, int* value) {
    int found[2] = {0, 0};
    if (mpi.rank == 0) {
        found[0] = nc_get_att_int(ncid, NC_GLOBAL, name.c_str(), &found[1]) == NC_NOERR;
    }
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        MPI_Bcast(found, 2, MPI_INT, 0, comm);
    }
#endif
    if (found[0]) {
        *value = found[1];
    }
    return found[0];
}

template <typename T>
std::vector<T> ScInFile::read_whole(const std::string& name) {
    int varid = -1;
    const auto s = shape_of(name, &varid);
    if (s.size() == 1 && s[0] == -1) {
        throw std::runtime_error(fmt::format("SC I/O: no variable {}", name));
    }
    i64 total = 1;
    for (auto e : s) {
        total *= e;
    }
    std::vector<T> result(total);
    if (mpi.rank == 0) {
        const std::vector<size_t> st(s.size(), 0), ct(s.begin(), s.end());
        nc_check(nc_get(ncid, varid, st.data(), ct.data(), result.data()), name);
    }
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        bcast(result.data(), total, comm);
    }
#endif
    return result;
}

template <typename T>
void ScInFile::read_block(const std::string& name, int split, i64 start, i64 count, T* out) {
    int varid = -1;
    const auto s = shape_of(name, &varid);
    if (s.empty() || (s.size() == 1 && s[0] == -1)) {
        throw std::runtime_error(fmt::format("SC I/O: no array variable {}", name));
    }
    std::vector<size_t> st, ct;
    const ScBlock mine(s, split, start, count, sizeof(T), chunk_bytes);
    if (mpi.num_ranks == 1) {
        for (i64 c = 0; c < mine.num_chunks(); ++c) {
            mine.hyperslab(c, &st, &ct);
            nc_check(nc_get(ncid, varid, st.data(), ct.data(), out + c * mine.rows_per_chunk * mine.row_elems), name);
        }
        return;
    }
#ifdef HAVE_MPI
    // NOTE(cmo): Rank 0 reads each rank's block in turn, sending a chunk while
    // reading the next.
    i64 mine_sc[2] = {start, count};
    std::vector<i64> all(2 * mpi.num_ranks);
    MPI_Gather(mine_sc, 2, MPI_INT64_T, all.data(), 2, MPI_INT64_T, 0, comm);
    const MPI_Datatype type = mpi_type_of<T>();
    if (mpi.rank != 0) {
        for (i64 c = 0; c < mine.num_chunks(); ++c) {
            MPI_Recv(out + c * mine.rows_per_chunk * mine.row_elems, int(mine.chunk_rows(c) * mine.row_elems), type, 0, 0, comm, MPI_STATUS_IGNORE);
        }
        return;
    }
    std::vector<T> buf[2];
    MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    for (int r = 0; r < mpi.num_ranks; ++r) {
        const ScBlock b(s, split, all[2 * r], all[2 * r + 1], sizeof(T), chunk_bytes);
        for (i64 c = 0; c < b.num_chunks(); ++c) {
            b.hyperslab(c, &st, &ct);
            if (r == 0) {
                nc_check(nc_get(ncid, varid, st.data(), ct.data(), out + c * b.rows_per_chunk * b.row_elems), name);
                continue;
            }
            auto& dst = buf[c % 2];
            MPI_Wait(&req[c % 2], MPI_STATUS_IGNORE);
            dst.resize(b.chunk_rows(c) * b.row_elems);
            nc_check(nc_get(ncid, varid, st.data(), ct.data(), dst.data()), name);
            MPI_Isend(dst.data(), int(dst.size()), type, r, 0, comm, &req[c % 2]);
        }
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }
#endif
}

int sc_nc_open_replicated(const MpiState& mpi, const std::string& path, int* ncid, std::vector<char>* buffer) {
    if (mpi.num_ranks == 1) {
        return nc_open(path.c_str(), NC_NOWRITE, ncid);
    }
#ifdef HAVE_MPI
    i64 size = -1;
    if (mpi.rank == 0) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (f) {
            size = f.tellg();
            buffer->resize(size);
            f.seekg(0);
            f.read(buffer->data(), size);
            if (!f) {
                size = -1;
            }
        }
    }
    MPI_Bcast(&size, 1, MPI_INT64_T, 0, mpi.comm);
    if (size < 0) {
        return NC_ENOTFOUND;
    }
    buffer->resize(size);
    bcast(buffer->data(), size, mpi.comm);
    return nc_open_mem(path.c_str(), NC_NOWRITE, size, buffer->data(), ncid);
#else
    return NC_ENOTFOUND;
#endif
}

#define SC_IO_INSTANTIATE(T) \
    template void ScOutFile::put_att<T>(const std::string&, const std::vector<T>&); \
    template void ScOutFile::write_block<T>(const std::string&, const std::vector<std::string>&, const std::vector<i64>&, int, i64, i64, const T*); \
    template std::vector<T> ScInFile::read_whole<T>(const std::string&); \
    template void ScInFile::read_block<T>(const std::string&, int, i64, i64, T*);
SC_IO_INSTANTIATE(f32)
SC_IO_INSTANTIATE(f64)
SC_IO_INSTANTIATE(i32)
SC_IO_INSTANTIATE(i64)
SC_IO_INSTANTIATE(u8)
SC_IO_INSTANTIATE(u16)
#undef SC_IO_INSTANTIATE
