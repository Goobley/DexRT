#if !defined(DEXRT_WAVELENGTH_BINS_HPP)
#define DEXRT_WAVELENGTH_BINS_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>

template <typename U>
struct TransitionWaveRange {
    int blue_idx;
    int red_idx;
    U blue;
    U red;
    bool continuum;
};

template <typename U>
struct WavelengthBins {
    std::vector<U> bin;
    /// Wavelengths in gaps between disjoint transitions where something else
    /// is active, but no sample: a sample must be added here.
    std::vector<U> holes;
};

/// Wavelength weights for the global grid. Midpoint method, except that governing
/// transition's edge bins are bounded, so they don't extend to a distant
/// neighbouring sample. In that case, we move the edges around. `governing` is
/// the index of the governing transition at each wavelength
template <typename U>
WavelengthBins<U> clamped_wavelength_bins(
    const std::vector<U>& grid,
    const std::vector<TransitionWaveRange<U>>& trans,
    const std::vector<int>& governing
) {
    const int n = grid.size();
    WavelengthBins<U> result;
    result.bin.resize(n);
    if (n < 2) {
        throw std::runtime_error("Wavelength grids need at least 2 points.");
    }

    // NOTE(cmo): Edge i must be between grid[i-1] and grid[i].
    std::vector<U> edge_min(n + 1);
    std::vector<U> edge_max(n + 1);
    // NOTE(cmo): The setter is the transition responsible for shifting an edge, -1 => none, sentinel
    std::vector<int> min_setter(n + 1, -1);
    std::vector<int> max_setter(n + 1, -1);
    // NOTE(cmo): The blue and red ends of the _global_ grid are fixed at the end sample points, we don't adjust these edges.
    for (int i = 1; i < n; ++i) {
        edge_min[i] = grid[i - 1];
        edge_max[i] = grid[i];
    }
    // NOTE(cmo): Governing lines have their edge bins limited to match the
    // interior half of this bin, whilst governing continua have their red ends
    // limited to the edge of the grid, no changes on their blue end
    for (int t = 0; t < trans.size(); ++t) {
        const auto& tr = trans[t];
        if (tr.red_idx - tr.blue_idx < 2) {
            // NOTE(cmo): If there's only one wavelength sample, leave it alone -- you need to know what you're doing.
            continue;
        }
        const int b = tr.blue_idx;
        const int r = tr.red_idx - 1;
        if (b > 0 && !tr.continuum && governing[b] == t) {
            const U lim = grid[b] - U(0.5) * (grid[b + 1] - grid[b]);
            if (lim > edge_min[b]) {
                edge_min[b] = lim;
                min_setter[b] = t;
            }
        }
        if (r + 1 < n && governing[r] == t) {
            U lim = grid[r];
            if (!tr.continuum) {
                lim += U(0.5) * (grid[r] - grid[r - 1]);
            }
            if (lim < edge_max[r + 1]) {
                edge_max[r + 1] = lim;
                max_setter[r + 1] = t;
            }
        }
    }

    auto any_active = [&trans](U wave) {
        for (const auto& tr : trans) {
            if (wave >= tr.blue && wave <= tr.red) {
                return true;
            }
        }
        return false;
    };

    // NOTE(cmo): Compute the new bin widths, and flag any holes
    // Bin i, around grid[i], spans [bin_blue[i], bin_red[i]]
    std::vector<U> bin_blue(n);
    std::vector<U> bin_red(n);
    bin_blue[0] = grid[0];
    bin_red[n - 1] = grid[n - 1];
    for (int i = 1; i < n; ++i) {
        // NOTE(cmo): Any edge change enforced by above is compatible (range hasn't flipped)
        if (edge_min[i] <= edge_max[i]) {
            const U edge = std::clamp(U(0.5) * (grid[i - 1] + grid[i]), edge_min[i], edge_max[i]);
            bin_red[i - 1] = edge;
            bin_blue[i] = edge;
            continue;
        }
        // NOTE(cmo): A transition ends at grid[i-1], another starts at grid[i],
        // and the intervening space can't be attached to either sample.
        const auto& ending = trans[max_setter[i]];
        const auto& starting = trans[min_setter[i]];
        // NOTE(cmo): Is there actually a gap between the transitions' specified
        // red and blue wavelengths from the model?
        const bool disjoint = ending.red < starting.blue;
        const U gap_mid = U(0.5) * (edge_max[i] + edge_min[i]);
        if (disjoint && !any_active(gap_mid)) {
            // There is a gap, and no known sources of opacity. Gap will be ignored.
            bin_red[i - 1] = edge_max[i];
            bin_blue[i] = edge_min[i];
            continue;
        }
        // Otherwise transitions overlap or something else is present in the
        // gap: split the difference, but flag the presence of a hole if
        // disjoint, and hope the caller refines.
        bin_red[i - 1] = gap_mid;
        bin_blue[i] = gap_mid;
        if (disjoint) {
            result.holes.emplace_back(gap_mid);
        }
    }
    for (int i = 0; i < n; ++i) {
        result.bin[i] = bin_red[i] - bin_blue[i];
    }
    return result;
}

#else
#endif
