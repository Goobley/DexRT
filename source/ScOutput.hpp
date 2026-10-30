#ifndef DEXRT_SC_OUTPUT_HPP
#define DEXRT_SC_OUTPUT_HPP

#include "ScCells.hpp"
#include "ScDecomposition.hpp"
#include "ScIo.hpp"
#include <string>
#include <vector>

/// Dimensions of a dense SC field [lead..., z, (y,) x], with its global
/// extents.
struct ScDenseLayout {
    std::vector<std::string> dims;
    std::vector<i64> shape;
    int split; /// index of z
};

inline ScDenseLayout sc_dense_layout(
    const ScCells& cells,
    const ScSlab& slab,
    const std::vector<std::string>& lead_dims,
    const std::vector<i64>& lead_shape
) {
    ScDenseLayout result{.dims = lead_dims, .shape = lead_shape, .split = int(lead_dims.size())};
    result.dims.push_back("z");
    result.shape.push_back(slab.global_num_z);
    if (cells.num_y > 1) {
        result.dims.push_back("y");
        result.shape.push_back(cells.num_y);
    }
    result.dims.push_back("x");
    result.shape.push_back(cells.num_x);
    return result;
}

/// Writes this rank's owned rows of SC fields, dense [lead..., z, (y,) x],
/// from the slab index space ([lead, ks], see ScCells). Void cells are zero.
class ScOutputWriter {
    ScOutFile& file;
    const ScSlab& slab;
    const ScCells& cells;

public:
    ScOutputWriter(ScOutFile& file_, const ScSlab& slab_, const ScCells& cells_)
        : file(file_), slab(slab_), cells(cells_) {}

    void write_active() {
        const auto active = cells.active.createHostCopy();
        const auto l = sc_dense_layout(cells, slab, {}, {});
        file.write_block("active", l.dims, l.shape, l.split, cells.z_start, cells.num_z, active.data() + cells.owned_offset());
    }

    /// A dense host field [lead, z_local, plane].
    template <typename T>
    void write_dense(const yakl::Array<T, 3, yakl::memHost>& local, const std::string& name, const std::string& lead) {
        const auto l = sc_dense_layout(cells, slab, {lead}, {i64(local.extent(0))});
        file.write_block(name, l.dims, l.shape, l.split, cells.z_start, cells.num_z, local.data());
    }

    /// [lead..., ks] (lead extents lead_shape).
    template <typename T>
    void write_ks(
        const yakl::Array<T, 2, yakl::memDevice>& array,
        const std::string& name,
        const std::vector<std::string>& lead_dims,
        const std::vector<i64>& lead_shape
    ) {
        const auto rows = cells.owned_rows(array);
        const auto l = sc_dense_layout(cells, slab, lead_dims, lead_shape);
        file.write_block(name, l.dims, l.shape, l.split, cells.z_start, cells.num_z, rows.data());
    }

    /// An owned-rows array [lead, z_local * plane] (J).
    void write_owned(const Rad16ArrayHost& array, const std::string& name, const std::string& lead) {
        write_dense(array.decode_rows(0, array.extent(0)).reshape(array.extent(0), cells.num_z, cells.plane_size()), name, lead);
    }

    void write_cells(const Fp2d& array, const std::string& name, const std::string& lead) {
        write_ks(array, name, {lead}, {i64(array.extent(0))});
    }

    template <typename T>
    void write_tensor(const yakl::Array<T, 3, yakl::memDevice>& array,
        const std::string& name,
        const std::string& dim0,
        const std::string& dim1) {
        const i32 n0 = array.extent(0), n1 = array.extent(1);
        write_ks(array.reshape(n0 * n1, cells.num_cells()), name, {dim0, dim1}, {n0, n1});
    }

    void write_scalar(const Fp1d& array, const std::string& name) {
        write_ks(array.reshape(1, cells.num_cells()), name, {}, {});
    }
};

#endif
