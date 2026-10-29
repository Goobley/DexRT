#if !defined(DEXRT_RAD16_OUTPUT_HPP)
#define DEXRT_RAD16_OUTPUT_HPP

#include "Rad16BitStorage.hpp"
#include "YAKL_netcdf.h"
#include <string>
#include <vector>

/// Rows of a Rad16ArrayHost decoded at a time when writing it out.
constexpr i64 RAD16_OUTPUT_CHUNK = 32;

/// Write a host [rows, ks] Rad16 array to nc as fp_t, RAD16_OUTPUT_CHUNK rows at
/// a time to bound the decoded copy. to_out maps the decoded rows (Fp2dHost
/// [chunk, ks]) to a host array in their output shape (leading dimension the
/// chunk), and dim_names names each output dimension.
template <typename ToOut>
void write_rad16_chunked(
    yakl::SimpleNetCDF& nc,
    const Rad16ArrayHost& arr,
    const std::string& name,
    const std::vector<std::string>& dim_names,
    const ToOut& to_out
) {
    using NcDim = yakl::SimpleNetCDF::NcDim;
    yakl::SimpleNetCDF::NcVar var;
    std::vector<size_t> start(dim_names.size(), 0);
    std::vector<size_t> count(dim_names.size(), 0);
    for (i64 row = 0; row < arr.extent(0); row += RAD16_OUTPUT_CHUNK) {
        const i64 num_rows = std::min(RAD16_OUTPUT_CHUNK, arr.extent(0) - row);
        const auto out = to_out(arr.decode_rows(row, num_rows));
        if (row == 0) {
            std::vector<NcDim> dims(dim_names.size());
            for (int i = 0; i < dim_names.size(); ++i) {
                const size_t len = (i == 0) ? arr.extent(0) : out.extent(i);
                dims[i] = nc.file.getDim(dim_names[i]);
                if (dims[i].isNull()) {
                    dims[i] = nc.file.addDim(dim_names[i], len);
                } else if (dims[i].getSize() != len) {
                    Kokkos::abort("dimension size differs from the file");
                }
                count[i] = len;
            }
            var = nc.file.addVar(name, std::is_same_v<fp_t, f64> ? NC_DOUBLE : NC_FLOAT, dims);
        }
        start[0] = row;
        count[0] = num_rows;
        var.putVar(start, count, out.data());
    }
}

#else
#endif
