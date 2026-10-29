#include "catch_amalgamated.hpp"
#include "WavelengthBins.hpp"
#include <numeric>
#include <cmath>

using Catch::Matchers::WithinAbs;

namespace {
typedef TransitionWaveRange<double> BT;

std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = a + (b - a) * double(i) / double(n - 1);
    }
    return result;
}

/// Merge sample sets (no duplicates expected) and fill blue/red_idx.
std::vector<double> merge(const std::vector<std::vector<double>>& sets) {
    std::vector<double> grid;
    for (const auto& s : sets) {
        grid.insert(grid.end(), s.begin(), s.end());
    }
    std::sort(grid.begin(), grid.end());
    return grid;
}

BT transition(const std::vector<double>& grid, double blue, double red, bool continuum) {
    return BT{
        .blue_idx = int(std::lower_bound(grid.begin(), grid.end(), blue) - grid.begin()),
        .red_idx = int(std::upper_bound(grid.begin(), grid.end(), red) - grid.begin()),
        .blue = blue,
        .red = red,
        .continuum = continuum
    };
}

/// Governing transition per sample: active lines first, then the continuum with
/// the closest red edge (as to_atomic_data, using the line centre for lines).
std::vector<int> governing(const std::vector<double>& grid, const std::vector<BT>& trans) {
    std::vector<int> result(grid.size(), -1);
    for (int i = 0; i < grid.size(); ++i) {
        double best = 1e30;
        bool have_line = false;
        for (int t = 0; t < trans.size(); ++t) {
            const auto& tr = trans[t];
            if (grid[i] < tr.blue || grid[i] > tr.red) {
                continue;
            }
            const bool line = !tr.continuum;
            const double dist = std::abs(grid[i] - (line ? 0.5 * (tr.blue + tr.red) : tr.red));
            if ((line && !have_line) || (line == have_line && dist < best)) {
                best = dist;
                have_line = have_line || line;
                result[i] = t;
            }
        }
    }
    return result;
}

WavelengthBins<double> bins_for(const std::vector<double>& grid, const std::vector<BT>& trans) {
    return clamped_wavelength_bins(grid, trans, governing(grid, trans));
}

int idx_of(const std::vector<double>& grid, double w) {
    return std::lower_bound(grid.begin(), grid.end(), w - 1e-9) - grid.begin();
}

double sum(const std::vector<double>& x) {
    return std::accumulate(x.begin(), x.end(), 0.0);
}
}

TEST_CASE("Clamped wavelength bins", "[wavelength_bins]") {
    SECTION("Isolated line in a coarse continuum") {
        // continuum 300..700 every 50 (without 500), line 495..505 every 1
        auto cont = linspace(300.0, 700.0, 9);
        cont.erase(cont.begin() + 4);
        const auto line = linspace(495.0, 505.0, 11);
        const auto grid = merge({cont, line});
        const std::vector<BT> trans{
            transition(grid, 495.0, 505.0, false),
            transition(grid, 300.0, 700.0, true)
        };
        const auto bins = bins_for(grid, trans);
        REQUIRE(bins.holes.empty());
        REQUIRE_THAT(sum(bins.bin), WithinAbs(400.0, 1e-10));
        // line end bins: half step inside + half step overhang
        REQUIRE_THAT(bins.bin[idx_of(grid, 495.0)], WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(bins.bin[idx_of(grid, 505.0)], WithinAbs(1.0, 1e-12));
        // the neighbours take the gap
        REQUIRE_THAT(bins.bin[idx_of(grid, 450.0)], WithinAbs(494.5 - 425.0, 1e-12));
        REQUIRE_THAT(bins.bin[idx_of(grid, 550.0)], WithinAbs(575.0 - 505.5, 1e-12));
        // interior unchanged trapezoid
        REQUIRE_THAT(bins.bin[idx_of(grid, 500.0)], WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(bins.bin[idx_of(grid, 350.0)], WithinAbs(50.0, 1e-12));
    }

    SECTION("Continuum red edge has no overhang") {
        const auto edge_cont = linspace(100.0, 200.0, 11);
        const std::vector<double> background{250.0, 300.0, 350.0, 400.0};
        const auto grid = merge({edge_cont, background});
        const std::vector<BT> trans{
            transition(grid, 100.0, 200.0, true),
            transition(grid, 100.0, 400.0, true)
        };
        const auto bins = bins_for(grid, trans);
        REQUIRE(bins.holes.empty());
        REQUIRE_THAT(sum(bins.bin), WithinAbs(300.0, 1e-10));
        REQUIRE_THAT(bins.bin[idx_of(grid, 200.0)], WithinAbs(5.0, 1e-12));
        REQUIRE_THAT(bins.bin[idx_of(grid, 250.0)], WithinAbs(275.0 - 200.0, 1e-12));
    }

    SECTION("Chained continua share an edge sample") {
        // A: 100..200, B: 200..400, sharing the sample at 200
        const auto ca = linspace(100.0, 200.0, 11);
        const std::vector<double> cb{250.0, 300.0, 350.0, 400.0};
        const auto grid = merge({ca, cb});
        const std::vector<BT> trans{
            transition(grid, 100.0, 200.0, true),
            transition(grid, 200.0, 400.0, true)
        };
        const auto bins = bins_for(grid, trans);
        REQUIRE(bins.holes.empty());
        // A's hard red edge; B's blue end at the midpoint
        REQUIRE_THAT(bins.bin[idx_of(grid, 200.0)], WithinAbs(5.0, 1e-12));
        REQUIRE_THAT(bins.bin[idx_of(grid, 250.0)], WithinAbs(275.0 - 200.0, 1e-12));
        REQUIRE_THAT(sum(bins.bin), WithinAbs(300.0, 1e-10));
    }

    SECTION("Continuum edge inside a line leaves the line's bins alone") {
        // continuum A ends at 121.6 inside the line's range, its own samples
        // pruned there; the line's sample at 121.5 is A's last.
        const std::vector<double> background{90.0, 130.0};
        const std::vector<double> a{100.0, 110.0};
        const auto line = linspace(120.0, 123.0, 13);
        const auto grid = merge({background, a, line});
        const std::vector<BT> trans{
            transition(grid, 120.0, 123.0, false),
            transition(grid, 100.0, 121.6, true),
            transition(grid, 90.0, 130.0, true)
        };
        const auto bins = bins_for(grid, trans);
        REQUIRE(bins.holes.empty());
        REQUIRE_THAT(bins.bin[idx_of(grid, 121.5)], WithinAbs(0.25, 1e-12));
        REQUIRE_THAT(bins.bin[idx_of(grid, 121.75)], WithinAbs(0.25, 1e-12));
        REQUIRE_THAT(sum(bins.bin), WithinAbs(40.0, 1e-10));
    }

    SECTION("Overlapping lines without a sample in the overlap") {
        // A ends at 399 (model red 399.7), B starts at 400 (model blue 399.3),
        // over a background continuum.
        const std::vector<double> a{397.8, 398.2, 398.6, 399.0};
        const std::vector<double> b{400.0, 400.4, 400.8, 401.2};
        const std::vector<double> background{390.0, 410.0};
        const auto grid = merge({a, b, background});
        const std::vector<BT> trans{
            transition(grid, 397.8, 399.7, false),
            transition(grid, 399.3, 401.2, false),
            transition(grid, 390.0, 410.0, true)
        };
        const auto bins = bins_for(grid, trans);
        REQUIRE(bins.holes.empty());
        REQUIRE_THAT(sum(bins.bin), WithinAbs(20.0, 1e-10));
        // limits 399.2 and 399.8 conflict: the edge splits them
        REQUIRE_THAT(bins.bin[idx_of(grid, 399.0)], WithinAbs(399.5 - 398.8, 1e-12));
        REQUIRE_THAT(bins.bin[idx_of(grid, 400.0)], WithinAbs(400.2 - 399.5, 1e-12));
    }

    SECTION("Disjoint lines with a background gap") {
        const std::vector<double> a{397.8, 398.2, 398.6, 399.0};
        const std::vector<double> b{400.0, 400.4, 400.8, 401.2};
        const std::vector<double> background{390.0, 410.0};
        {
            const auto grid = merge({a, b, background});
            const std::vector<BT> trans{
                transition(grid, 397.8, 399.0, false),
                transition(grid, 400.0, 401.2, false),
                transition(grid, 390.0, 410.0, true)
            };
            const auto bins = bins_for(grid, trans);
            REQUIRE(bins.holes.size() == 1);
            REQUIRE_THAT(bins.holes[0], WithinAbs(399.5, 1e-12));

            // filling the hole resolves it, with the gap shared by the new sample
            auto filled = grid;
            filled.insert(filled.begin() + idx_of(filled, 399.5), 399.5);
            const std::vector<BT> trans_filled{
                transition(filled, 397.8, 399.0, false),
                transition(filled, 400.0, 401.2, false),
                transition(filled, 390.0, 410.0, true)
            };
            const auto bins_filled = bins_for(filled, trans_filled);
            REQUIRE(bins_filled.holes.empty());
            REQUIRE_THAT(sum(bins_filled.bin), WithinAbs(20.0, 1e-10));
            REQUIRE_THAT(bins_filled.bin[idx_of(filled, 399.0)], WithinAbs(0.4, 1e-12));
            REQUIRE_THAT(bins_filled.bin[idx_of(filled, 399.5)], WithinAbs(399.8 - 399.2, 1e-12));
        }
        {
            // no background: nothing to integrate in the gap, no hole
            const auto grid = merge({a, b});
            const std::vector<BT> trans{
                transition(grid, 397.8, 399.0, false),
                transition(grid, 400.0, 401.2, false)
            };
            const auto bins = bins_for(grid, trans);
            REQUIRE(bins.holes.empty());
            // the gap is in neither bin
            REQUIRE_THAT(bins.bin[idx_of(grid, 399.0)], WithinAbs(0.4, 1e-12));
            REQUIRE_THAT(bins.bin[idx_of(grid, 400.0)], WithinAbs(0.4, 1e-12));
            REQUIRE_THAT(sum(bins.bin), WithinAbs(401.2 - 397.8 - (399.8 - 399.2), 1e-10));
        }
    }

    SECTION("Bins positive, sum to span") {
        // dense lines in a sparse continuum, lines touching the grid ends
        const auto cont = linspace(10.0, 1000.0, 12);
        const auto l0 = linspace(10.5, 12.0, 4);
        const auto l1 = linspace(500.1, 501.0, 10);
        const auto l2 = linspace(999.0, 999.9, 4);
        const auto grid = merge({cont, l0, l1, l2});
        const std::vector<BT> trans{
            transition(grid, 10.5, 12.0, false),
            transition(grid, 500.1, 501.0, false),
            transition(grid, 999.0, 999.9, false),
            transition(grid, 10.0, 1000.0, true)
        };
        const auto bins = bins_for(grid, trans);
        REQUIRE(bins.holes.empty());
        REQUIRE_THAT(sum(bins.bin), WithinAbs(990.0, 1e-9));
        for (const double b : bins.bin) {
            REQUIRE(b > 0.0);
        }
    }
}
