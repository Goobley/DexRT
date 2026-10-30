#ifndef DEXRT_SC_HALO_HPP
#define DEXRT_SC_HALO_HPP
#include "ScState.hpp"
#include <vector>

/// The rows of a dense field in the ScCells layout: element
/// (l, z, j) at data[l * lead_stride + (z + SC_MEDIUM_HALO) * row_size + j],
/// local z, j in [0, row_size).
struct ScHaloField {
    void* data;
    i64 lead;
    i64 lead_stride;
    i64 row_size;
    i32 bytes; /// per element: 1, 2, 4 or 8
};

/// Fills the halo rows of this rank's dense fields from the ranks that own
/// them. Each halo row comes directly from its owner, so slabs thinner than
/// the halo work. Rows beyond the domain repeat the edge row (zero gradient,
/// as the sweep's clamp on a single rank).
class ScHalo {
    struct Peer {
        i32 rank;
        yakl::Array<i32, 1, yakl::memDevice> send_rows; /// local owned rows, in message order
        yakl::Array<i32, 1, yakl::memDevice> recv_rows; /// local halo rows, in message order
        std::vector<i32> send_host, recv_host;
    };
    std::vector<Peer> peers;
    /// Halo rows whose (clamped) source row is our own.
    yakl::Array<i32, 1, yakl::memDevice> self_src, self_dst;
    std::vector<i32> self_src_host, self_dst_host;
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_NULL;
#endif

    /// Host: the halo rows of an owned-rows array ([lead, z * plane], e.g.
    /// J_cpu) into halo ([lead, h * plane], h = z + H below the slab, z -
    /// num_z + H above).
    void exchange_halo_rows_host(const Rad16ArrayHost& owned, const Rad16ArrayHost& halo, i32 num_z, i64 plane) const;

public:
    ~ScHalo();
    /// Collective.
    void init(const ScState& state);
    /// Collective; fields must match on every rank.
    void exchange(const std::vector<ScHaloField>& fields) const;
    /// The halo rows' activity, rebuilding state->cells' lists (setup), and
    /// the GivenFs emissivity/opacity.
    void exchange_static(ScState* state) const;
    /// All inputs of a formal solution: medium halos (atmosphere, populations,
    /// PRD rho* and background) and previous-J halos for background scattering.
    /// Call after owned nh0/background updates, before the solve overwrites J.
    void exchange_inputs(ScState* state) const;
};

#endif
