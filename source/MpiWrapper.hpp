#if !defined(DEXRT_MPI_WRAPPER_HPP)
#define DEXRT_MPI_WRAPPER_HPP

#ifdef HAVE_MPI
#include <mpi.h>
#include <fmt/core.h>
#endif

inline void init_mpi(int argc, char* argv[]) {
#ifdef HAVE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fmt::println("Insufficient MPI threading support, got {}", provided);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
#endif
}

inline void finalise_mpi() {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}

struct MpiState {
    int rank = 0;
    int num_ranks = 1;
#ifdef HAVE_MPI
    MPI_Comm comm;
#endif
};

#else
#endif