#if !defined(DEXRT_WAVELENGTH_PARALLELISATION_HPP)
#define DEXRT_WAVELENGTH_PARALLELISATION_HPP

#include "MpiWrapper.hpp"
#include "State.hpp"

#include <mutex>
#include <optional>

template <typename State>
inline void setup_comm(State* state) {
#ifdef HAVE_MPI
    MPI_Comm_dup(MPI_COMM_WORLD, &state->mpi_state.comm);
    MPI_Comm_size(state->mpi_state.comm, &state->mpi_state.num_ranks);
    MPI_Comm_rank(state->mpi_state.comm, &state->mpi_state.rank);
    MPI_Barrier(state->mpi_state.comm);
    state->println("Running on {} ranks", state->mpi_state.num_ranks);
#endif
}

inline bool should_write(State* state) {
#ifdef HAVE_MPI
    return state->mpi_state.rank == 0;
#endif
    return true;
}

struct WavelengthBatch {
    int la_start;
    int la_end;
};

struct WavelengthDistributor {
    int la = 0;
    int la_max;
    int batch_size;
    int rank_la_start = 0;
#ifdef HAVE_MPI
    std::jthread service_thread;
    std::mutex lock;
    std::mutex test_lock; // For protecting the load-balancing test
#endif

    static constexpr int wavelength_tag = 1001;
    static constexpr int req_size = 2;

    ~WavelengthDistributor() {
#ifdef HAVE_MPI
        if constexpr (MPI_LOAD_BALANCE) {
            service_thread.request_stop();
        }
#endif
    }

    inline bool init(const MpiState& mpi_state, int la_max_, int batch_size_) {
        la = 0; // only used on rank 0 or if not load balancing
        la_max = la_max_;
        batch_size = batch_size_;
    #ifdef HAVE_MPI
        if constexpr (MPI_LOAD_BALANCE) {
            if (mpi_state.rank == 0) {
                service_thread = std::jthread(
                        [this, mpi_state] (std::stop_token stopper) {
                            this->serve_requests(stopper, mpi_state);
                        }
                    );
            }
        } else {
            rank_la_start = mpi_state.rank * batch_size;
        }
    #endif
        return true;
    }

    inline void reset() {
#ifdef HAVE_MPI
        std::lock_guard<std::mutex> lock_holder(lock);
#endif
        la = rank_la_start;
    }

#ifdef HAVE_MPI
    inline int get_next_la_critical(int this_batch_size) {
        std::lock_guard<std::mutex> lock_holder(lock);
        int next_la = la;
        la += this_batch_size;
        return next_la;
    }

    inline void serve_requests(std::stop_token stop_token, const MpiState& mpi_state) {
        int from[req_size];

        auto loop_body = [&]() {
            int message_ready = 0;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, wavelength_tag, mpi_state.comm, &message_ready, &status);
            if (message_ready) {
                MPI_Recv(from, req_size, MPI_INT, MPI_ANY_SOURCE, wavelength_tag, mpi_state.comm, &status);
                int rank_from = from[0];
                int rank_batch_size = from[1];
                int next_la = get_next_la_critical(rank_batch_size);
                MPI_Send(&next_la, 1, MPI_INT, status.MPI_SOURCE, wavelength_tag, mpi_state.comm);
            }
        };

        while (!stop_token.stop_requested()) {
            if constexpr (MPI_PROTECT_TEST) {
                std::lock_guard<std::mutex> lock_holder(test_lock);
                loop_body();
            } else {
                loop_body();
            }
        }
    }

    inline int request_next_la(const MpiState& mpi_state) {
        int send_packet[req_size] = {mpi_state.rank, batch_size};
        MPI_Send(send_packet, req_size, MPI_INT, 0, wavelength_tag, mpi_state.comm);
        int next_la = 0;
        MPI_Recv(&next_la, 1, MPI_INT, 0, wavelength_tag, mpi_state.comm, MPI_STATUS_IGNORE);
        return next_la;
    }
#endif

    inline bool next_batch(const MpiState& mpi_state, WavelengthBatch* batch) {
#ifdef HAVE_MPI
        int next_la;
        if constexpr (MPI_LOAD_BALANCE) {
            if (mpi_state.rank == 0) {
                next_la = get_next_la_critical(batch_size);
            } else {
                next_la = request_next_la(mpi_state);
            }
        } else {
            // Get next wavelength, step with stride num_ranks * batch_size
            next_la = la;
            la += mpi_state.num_ranks * batch_size;
        }
        batch->la_start = next_la;
        batch->la_end = std::min(batch->la_start + batch_size, la_max);
        return batch->la_start < la_max;
#else
        batch->la_start = la;
        batch->la_end = std::min(la + batch_size, la_max);
        la += batch_size;
        return batch->la_start < la_max;
#endif
    }

    inline void wait_for_all(const MpiState& mpi_state) {
#ifdef HAVE_MPI
        MPI_Barrier(mpi_state.comm);
#endif
    }

    template <typename State>
    inline void reduce_Gamma(State* state) {
#ifdef HAVE_MPI
        std::lock_guard<std::mutex> lock_holder(test_lock);
        for (int ia = 0; ia < state->Gamma.size(); ++ia) {
            if (state->mpi_state.rank == 0) {
                MPI_Reduce(
                    MPI_IN_PLACE,
                    state->Gamma[ia].data(),
                    state->Gamma[ia].size(),
                    get_GammaFpMpi(),
                    MPI_SUM,
                    0,
                    state->mpi_state.comm
                );
            } else {
                MPI_Reduce(
                    state->Gamma[ia].data(),
                    state->Gamma[ia].data(),
                    state->Gamma[ia].size(),
                    get_GammaFpMpi(),
                    MPI_SUM,
                    0,
                    state->mpi_state.comm
                );
            }
        }
#endif
    }

    template <typename State>
    inline void reduce_J(State* state) {
#ifdef HAVE_MPI
        std::lock_guard<std::mutex> lock_holder(test_lock);
        fp_t* J_ptr = state->config.store_J_on_cpu ? state->J_cpu.data() : state->J.data();
        i64 J_size = state->config.store_J_on_cpu ? state->J_cpu.size() : state->J.size();
        if (state->mpi_state.rank == 0) {
            MPI_Reduce(
                MPI_IN_PLACE,
                J_ptr,
                J_size,
                get_FpMpi(),
                MPI_SUM,
                0,
                state->mpi_state.comm
            );
        } else {
            MPI_Reduce(
                J_ptr,
                J_ptr,
                J_size,
                get_FpMpi(),
                MPI_SUM,
                0,
                state->mpi_state.comm
            );
        }
#endif
    }

    template <typename State>
    inline void reduce_rad_loss(State* state) {
#ifdef HAVE_MPI
        std::lock_guard<std::mutex> lock_holder(test_lock);
        RadLossFp* rad_loss_ptr = state->config.store_J_on_cpu ? state->rad_loss_cpu.data() : state->rad_loss.data();
        i64 rad_loss_size = state->config.store_J_on_cpu ? state->rad_loss_cpu.size() : state->rad_loss.size();
        if (state->mpi_state.rank == 0) {
            MPI_Reduce(
                MPI_IN_PLACE,
                rad_loss_ptr,
                rad_loss_size,
                get_RadLossFpMpi(),
                MPI_SUM,
                0,
                state->mpi_state.comm
            );
        } else {
            MPI_Reduce(
                rad_loss_ptr,
                rad_loss_ptr,
                rad_loss_size,
                get_RadLossFpMpi(),
                MPI_SUM,
                0,
                state->mpi_state.comm
            );
        }
#endif
    }

    template <typename State>
    inline void update_pops(State* state) {
#ifdef HAVE_MPI
        std::lock_guard<std::mutex> lock_holder(test_lock);
        MPI_Bcast(state->pops.data(), state->pops.size(), get_FpMpi(), 0, state->mpi_state.comm);
#endif
    }

    template <typename State>
    inline void update_ne(State* state) {
#ifdef HAVE_MPI
        std::lock_guard<std::mutex> lock_holder(test_lock);
        MPI_Bcast(state->atmos.ne.data(), state->atmos.ne.size(), get_FpMpi(), 0, state->mpi_state.comm);
#endif
    }

    template <typename State>
    inline void update_nh_tot(State* state) {
#ifdef HAVE_MPI
        std::lock_guard<std::mutex> lock_holder(test_lock);
        MPI_Bcast(state->atmos.nh_tot.data(), state->atmos.nh_tot.size(), get_FpMpi(), 0, state->mpi_state.comm);
#endif
    }

};



#else
#endif