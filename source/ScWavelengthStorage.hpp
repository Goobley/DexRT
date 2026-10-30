#ifndef DEXRT_SC_WAVELENGTH_STORAGE_HPP
#define DEXRT_SC_WAVELENGTH_STORAGE_HPP
#include "ScState.hpp"
#include "ScConfig.hpp"
#include "ScDynamicScheduler.hpp"
#include <algorithm>
#include <vector>

/// Scheduler admission reserves a slot across every direction of a wavelength
/// batch. The payloads belong here; task scratch and MPI rows have other owners.
struct ScWavelengthStorage {
    struct Slot {
        Fp2d J;
        Fp2d loss;
        Rad16Array previous_J;
    };
    std::vector<Slot> slots;
    Rad16Array J_stage;
    Rad16ArrayHost J_prev_stage;
    bool has_J = false;
    bool has_loss = false;
    bool background_enabled = false;
    bool store_J = false;
    bool store_loss = false;

    i32 slot_count() const {
        return store_J || store_loss || background_enabled ? slots.size() : 0;
    }
    Fp2d J(i32 slot) const { return store_J ? slots[slot].J : Fp2d(); }
    Fp2d loss(i32 slot) const { return store_loss ? slots[slot].loss : Fp2d(); }
    Rad16Array previous_J(i32 slot) const {
        return background_enabled ? slots[slot].previous_J : Rad16Array();
    }
};

/// Allocate wavelength payloads independently of task scratch and coefficient eviction.
inline ScWavelengthStorage sc_make_wavelength_storage(
    const ScState& state,
    i32 num_ranks,
    const ScOptions& opts,
    i32 wave_batch
) {
    ScWavelengthStorage storage{
        .has_J = state.J_cpu.initialized(),
        .has_loss = state.rad_loss_integrated.initialized(),
        .background_enabled = state.background.enabled,
    };
    if (!storage.has_J && !storage.has_loss && !storage.background_enabled) {
        return storage;
    }
    const i32 nwave = state.config.mode == DexrtMode::GivenFs
        ? state.given_state.emis.extent(2)
        : state.adata.wavelength.extent(0);
    const i32 num_batches = (nwave + wave_batch - 1) / wave_batch;
    const i32 count = opts.j_slots ? opts.j_slots : std::min(num_ranks + opts.receive_window, num_batches);
    const i64 owned = state.cells.num_owned_cells();
    storage.slots.resize(count);
    for (auto& slot : storage.slots) {
        if (storage.has_loss) {
            slot.loss = Fp2d("sc_rad_loss_slot", wave_batch, owned);
        }
        if (storage.has_J) {
            slot.J = Fp2d("sc_J_slot", wave_batch, owned);
        }
        if (storage.background_enabled) {
            slot.previous_J = Rad16Array("sc_J_prev_slot", wave_batch, state.cells.num_cells());
        }
    }
    if (HALF_RADIATION_STORAGE && storage.has_J) {
        storage.J_stage = Rad16Array("sc_J_stage", wave_batch, owned);
    }
    if (storage.background_enabled) {
        storage.J_prev_stage = Rad16ArrayHost("sc_J_prev_stage", wave_batch, state.cells.num_cells());
    }
    return storage;
}

/// PRD-only passes use the scattering snapshot without replacing full-solve sums.
inline void sc_begin_wavelength_solve(ScWavelengthStorage& storage, bool prd_only) {
    storage.store_J = storage.has_J && !prd_only;
    storage.store_loss = storage.has_loss && !prd_only;
}

/// Upload owned and halo rows before any direction can overwrite this wavelength's host J.
inline void sc_load_previous_J(
    ScWavelengthStorage& storage,
    const ScState& state,
    i32 slot,
    i32 la,
    i32 end
) {
    yakl::timer_start("SC J_prev upload");
    const auto& dst = storage.slots[slot].previous_J;
    if (!state.background.have_J) {
        dst.fill(FP(0.0));
    } else {
        using T = Rad16ArrayHost::Storage;
        const i64 off = state.cells.owned_offset();
        const i64 owned = state.cells.num_owned_cells();
        const i64 n = storage.J_prev_stage.extent(1);
        for (i32 r = 0; r < end - la; ++r) {
            T* row = storage.J_prev_stage.data.data() + r * n;
            const T* halo = state.J_halo_cpu.data.data() + (la + r) * 2 * off;
            std::copy_n(halo, off, row);
            std::copy_n(state.J_cpu.data.data() + (la + r) * owned, owned, row + off);
            std::copy_n(halo + off, off, row + off + owned);
        }
        storage.J_prev_stage.rows(0, end - la).deep_copy_to(dst.rows(0, end - la));
        Kokkos::fence();
    }
    yakl::timer_stop("SC J_prev upload");
}

/// Start admission with empty sums and an immutable scattering snapshot.
inline void sc_open_wavelength_slot(
    ScWavelengthStorage& storage,
    const ScState& state,
    const ScScheduledTask& task,
    i32 slot
) {
    if (storage.store_J) {
        storage.slots[slot].J = FP(0.0);
    }
    if (storage.store_loss) {
        storage.slots[slot].loss = FP(0.0);
    }
    if (storage.background_enabled) {
        sc_load_previous_J(storage, state, slot, task.la, task.end);
    }
}

/// Reduce wavelength loss and optionally publish its individual wavelength rows.
inline void sc_finalise_loss(const ScState& state, const Fp2d& slot, i32 la, i32 width) {
    const auto& integrated = state.rad_loss_integrated;
    dex_parallel_for(
        "SC integrate rad_loss batch",
        FlatLoop<1>(slot.extent(1)),
        YAKL_LAMBDA(i64 k) {
            RadLossFp sum = RadLossFp(0.0);
            for (i32 w = 0; w < width; ++w) {
                sum += slot(w, k);
            }
            integrated(k) += sum;
        }
    );
    if (state.config.rad_loss == RadLossType::PerWavelength) {
        const auto& host = state.rad_loss_cpu;
        Fp2d("sc_rad_loss_rows", slot.data(), width, slot.extent(1)).deep_copy_to(
            yakl::Array<RadLossFp, 2, yakl::memHost>(
                "sc_rad_loss_cpu_rows",
                host.data() + i64(la) * host.extent(1),
                width,
                host.extent(1)
            )
        );
    }
    Kokkos::fence();
}

/// Publish complete angular sums before the scheduler recycles the slot index.
inline void sc_close_wavelength_slot(
    ScWavelengthStorage& storage,
    const ScState& state,
    const ScScheduledTask& task,
    i32 slot
) {
    const i32 width = task.end - task.la;
    if (storage.store_J) {
        yakl::timer_start("SC J paging");
        page_rows_to_host(
            state.J_cpu,
            task.la,
            storage.slots[slot].J,
            width,
            storage.J_stage
        );
        yakl::timer_stop("SC J paging");
    }
    if (storage.store_loss) {
        sc_finalise_loss(state, storage.slots[slot].loss, task.la, width);
    }
}

/// Publish the integrated loss after all admitted batches have closed.
inline void sc_finish_wavelength_solve(const ScWavelengthStorage& storage, const ScState& state) {
    if (storage.store_loss && state.config.rad_loss == RadLossType::Integrated) {
        Fp2d(
            "sc_rad_loss_integrated",
            state.rad_loss_integrated.data(),
            1,
            state.rad_loss_integrated.extent(0)
        ).deep_copy_to(state.rad_loss_cpu);
        Kokkos::fence();
    }
}
#endif
