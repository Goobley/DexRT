#if !defined(DEXRT_POPULATIONS_HPP)
#define DEXRT_POPULATIONS_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "Constants.hpp"
#include "Utils.hpp"

template <typename T=fp_t, typename U=fp_t, int mem_space=memDevice>
CompAtom<T> to_comp_atom(const ModelAtom<U>& model) {
    CompAtom<T, yakl::memHost> host_atom;
    host_atom.mass = model.element.mass;
    host_atom.abundance = model.element.abundance;
    host_atom.Z = model.element.Z;

    const int n_level = model.levels.size();
    host_atom.energy = decltype(host_atom.energy)("energy", n_level);
    host_atom.g = decltype(host_atom.g)("g", n_level);
    host_atom.stage = decltype(host_atom.stage)("stage", n_level);

    for (int i = 0; i < n_level; ++i) {
        host_atom.energy(i) = model.levels[i].energy;
        host_atom.g(i) = model.g[i].energy;
        host_atom.stage(i) = model.stage[i].energy;
    }

    const int n_lines = model.lines.size();
    host_atom.lines = decltype(host_atom.lines)("lines", n_lines);
    int n_broadening = 0;
    int n_max_wavelengths = 0;
    for (const auto& line : model.lines) {
        n_broadening += line.broadening.size();
        n_max_wavelengths += line.wavelength.size();
    }
    host_atom.broadening = decltype(host_atom.broadening)("broadening", n_broadening);
    int broad_idx = 0;
    for (int kr = 0; kr < n_lines; ++kr) {
        const auto& l = model.lines[l];
        auto& new_l = host_atom.lines(kr);
        new_l.type = l.type;
        new_l.j = l.j;
        new_l.i = l.i;
        new_l.f = l.f;
        new_l.g_natural = l.g_natural;
        new_l.Aji = l.Aji;
        new_l.Bji = l.Bji;
        new_l.Bji_wavelength = l.Bji_wavelength;
        new_l.Bij = l.Bij;
        new_l.Bij_wavelength = l.Bij_wavelength;
        new_l.lambda0 = l.lambda0;

        new_l.broad_start = broad_idx;
        for (int local_b_idx = 0; local_b_idx < l.broadening.size(); ++local_b_idx) {
            host_atom.broadening(broad_idx) = l.broadening[local_b_idx];
            broad_idx += 1;
        }
        new_l.broad_end = broad_idx;
    }


    struct WavelengthRegime {
        U start;
        U end;
    }
    std::vector<WavelengthRegime> wavelength_bands;
    U bluest = FP(1e30);
    for (const auto& line : model.lines) {
        bluest = yakl::min(bluest, line.wavelength[0]);
    }
    for (const auto& cont : model.continua) {
        bluest = yakl::min(bluest, cont.wavelength[0]);
    }

    while (true) {
        for ()
    }





    if constexpr (mem_space == memDevice) {

    }
}


template <typename T=fp_t, int mem_space>
YAKL_INLINE
void lte_pops(
    const yakl::Array<fp_t const, 1, mem_space>& energy,
    const yakl::Array<int const, 1, mem_space>& g,
    const yakl::Array<int const, 1, mem_space>& stage,
    fp_t temperature,
    fp_t ne,
    fp_t ntot,
    yakl::Array<fp_t, 1, mem_space>& pops
) {
    using namespace ConstantsFP;
    constexpr fp_t debroglie_const = square(h) / (FP(2.0) * FP(M_PI) * m_e);

    const int n_level = energy.extent(0);

    const fp_t kbT = temperature * k_B_eV;
    const fp_t saha_term = 1.5 * ne * std::pow(debroglie_const / temperature, FP(1.5));
    fp_t sum = FP(1.0);

    for (int i = 1; i < n_level; ++i) {
        const fp_t dE = energy(i) - energy(0);
        const fp_t gi0 = g(i) / g(0);
        const int dZ = stage(i) - stage(0);

        const fp_t dE_kbT = dE / kbT;
        fp_t pop_i = gi0 * std::exp(-dE_kbT);
        for (int _ = 1; _ <= dZ; ++_) {
            pop_i /= saha_term;
        }
        sum += pop_i;
        pops(i) = pop_i;
    }
    const fp_t pop_0 = ntot / sum;
    pops(0) = pop_0;

    for (int i = 1; i < n_level; ++i) {
        pops(i) *= yakl::max(pop_0, std::numeric_limits<fp_t>::min());
    }
}

#else
#endif