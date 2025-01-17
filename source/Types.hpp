#if !defined(DEXRT_TYPES_HPP)
#define DEXRT_TYPES_HPP
#include "Config.hpp"
#include "Constants.hpp"

typedef Kokkos::LayoutRight Layout;
template <class T, typename... Args>
using KView = Kokkos::View<T, Layout, Args...>;

typedef Kokkos::DefaultExecutionSpace::memory_space DefaultMemSpace;
typedef Kokkos::HostSpace HostSpace;
constexpr bool HostDevSameSpace = std::is_same_v<DefaultMemSpace, HostSpace>;

typedef Kokkos::View<fp_t*    , Layout> Fp1d;
typedef Kokkos::View<fp_t**   , Layout> Fp2d;
typedef Kokkos::View<fp_t***  , Layout> Fp3d;
typedef Kokkos::View<fp_t**** , Layout> Fp4d;
typedef Kokkos::View<fp_t*****, Layout> Fp5d;

typedef Kokkos::View<const fp_t*    , Layout> FpConst1d;
typedef Kokkos::View<const fp_t**   , Layout> FpConst2d;
typedef Kokkos::View<const fp_t***  , Layout> FpConst3d;
typedef Kokkos::View<const fp_t**** , Layout> FpConst4d;
typedef Kokkos::View<const fp_t*****, Layout> FpConst5d;

typedef KView<GammaFp***, DefaultMemSpace> GammaMat;

typedef yakl::SArray<fp_t, 1, 2> vec2;
typedef yakl::SArray<fp_t, 1, 3> vec3;
typedef yakl::SArray<fp_t, 1, 4> vec4;
typedef yakl::SArray<fp_t, 2, 2, 2> mat2x2;
typedef yakl::SArray<int, 2, 2, 2> imat2x2;
typedef yakl::SArray<int32_t, 1, 2> ivec2;

struct Coord2 {
    i32 x;
    i32 z;

    YAKL_INLINE bool operator==(const Coord2& other) const {
        return (x == other.x) && (z == other.z);
    }
};

template <int R, typename... Args>
using MDRange = Kokkos::MDRangePolicy<Kokkos::Rank<R, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, Args...>;

using Kokkos::parallel_for;

// using yakl::c::parallel_for;
// using yakl::c::SimpleBounds;
// using yakl::Dims;
struct State;

enum class DexrtMode {
    Lte,
    NonLte,
    GivenFs,
};

/// The texel storage setup for a cascade
struct CascadeStorage {
    ivec2 num_probes;
    int num_flat_dirs;
    int wave_batch;
    int num_incl;
};

/// The rays to be computed in a cascade
struct CascadeRays {
    ivec2 num_probes;
    int num_flat_dirs;
    int wave_batch;
    int num_incl;
};

/// The rays in the partition of a cascade we're computing. For all
/// entries we compute [start, start+num)
struct CascadeRaysSubset {
    ivec2 start_probes;
    ivec2 num_probes;
    int start_flat_dirs;
    int num_flat_dirs;
    int start_wave_batch;
    int wave_batch;
    int start_incl;
    int num_incl;
};

struct InclQuadrature {
    Fp1d muy;
    Fp1d wmuy;
};

struct DexEmpty {};

template <typename Alo=DexEmpty>
struct RadianceInterval {
    fp_t I = FP(0.0);
    fp_t tau = FP(0.0);
    Alo alo;
};

struct RayProps {
    vec2 start;
    vec2 end;
    vec2 dir;
    vec2 centre;
};

struct DeviceProbesToCompute {
    bool sparse;
    ivec2 num_probes;
    KView<i32*[2]> active_probes; // [n, 2 (u, v)]

    YAKL_INLINE ivec2 operator()(i64 ks) const {
        if (!sparse) {
            // NOTE(cmo): As in the loop over probes we iterate as [v, u] (u
            // fast-running), but index as [u, v], i.e. dims.num_probes(0) =
            // dim(u). Typical definition of k = u * Nv + v, but here we do
            // loop index k = v * Nu + u where Nu = dims.num_probes(0). This
            // preserves our iteration ordering
            ivec2 probe_coord;
            probe_coord(0) = ks % num_probes(0);
            probe_coord(1) = ks / num_probes(0);
            return probe_coord;
        }

        ivec2 probe_coord;
        probe_coord(0) = active_probes(ks, 0);
        probe_coord(1) = active_probes(ks, 1);
        return probe_coord;
    }

    YAKL_INLINE i64 num_active_probes() const {
        if (sparse) {
            return active_probes.extent(0);
        }
        return num_probes(0) * num_probes(1);
    }
};

struct ProbesToCompute {
    bool sparse;
    CascadeStorage c0_size;
    std::vector<KView<i32*[2]>> active_probes; // [n, 2 (u, v)]

    /// Setup object, low-level
    void init(
        const CascadeStorage& c0,
        bool sparse,
        std::vector<KView<i32*[2]>> active_probes = decltype(active_probes)()
    );

    /// Setup object, high-level, from state
    void init(
        const State& state,
        int max_cascade
    );

    /// Bind cascade n to device type
    DeviceProbesToCompute bind(int cascade_idx) const;

    /// Number of probes to compute in cascade n
    i64 num_active_probes(int cascade_idx) const;
};

struct CascadeCalcSubset {
    int la_start = -1;
    int la_end = -1;
    int subset_idx = 0;
};

struct Atmosphere {
    fp_t voxel_scale;
    fp_t offset_x = FP(0.0);
    fp_t offset_y = FP(0.0);
    fp_t offset_z = FP(0.0);
    bool moving = false;
    Fp2d temperature;
    Fp2d pressure;
    Fp2d ne;
    Fp2d nh_tot;
    Fp2d nh0;
    Fp2d vturb;
    Fp2d vx;
    Fp2d vy;
    Fp2d vz;
};

struct SparseAtmosphere {
    fp_t voxel_scale;
    fp_t offset_x = FP(0.0);
    fp_t offset_y = FP(0.0);
    fp_t offset_z = FP(0.0);
    i32 num_x;
    i32 num_y;
    i32 num_z;
    bool moving = false;
    Fp1d temperature;
    Fp1d pressure;
    Fp1d ne;
    Fp1d nh_tot;
    Fp1d nh0;
    Fp1d vturb;
    Fp1d vx;
    Fp1d vy;
    Fp1d vz;
};

template <typename T=fp_t>
struct FlatAtmosphere {
    fp_t voxel_scale;
    fp_t offset_x = FP(0.0);
    fp_t offset_y = FP(0.0);
    fp_t offset_z = FP(0.0);
    bool moving = false;
    Kokkos::View<T*, DefaultMemSpace> temperature;
    Kokkos::View<T*, DefaultMemSpace> pressure;
    Kokkos::View<T*, DefaultMemSpace> ne;
    Kokkos::View<T*, DefaultMemSpace> nh_tot;
    Kokkos::View<T*, DefaultMemSpace> nh0;
    Kokkos::View<T*, DefaultMemSpace> vturb;
    Kokkos::View<T*, DefaultMemSpace> vx;
    Kokkos::View<T*, DefaultMemSpace> vy;
    Kokkos::View<T*, DefaultMemSpace> vz;
};

/// Storage for emissivity/opacity when we load a model from file. Planes from
/// this get copied into the normal buffer before solution via RC.
struct GivenEmisOpac {
    fp_t voxel_scale;
    Fp3d emis;
    Fp3d opac;
};

template <typename T=fp_t>
struct Element {
    std::string symbol;
    T mass;
    T abundance;
    int Z;
};

template <typename T=fp_t>
struct AtomicLevel {
    /// in eV
    T energy;
    /// statistical weight
    int g;
    /// ionisation stage
    int stage;
    std::string key;
    std::string label;
};

enum class LineProfileType {
    Voigt,
    PrdVoigt
};

template <typename T=fp_t>
struct ScaledExponentsBroadening {
    T scaling;
    T temperature_exponent;
    T hydrogen_exponent;
    T electron_exponent;
};

template <typename T=fp_t>
struct AtomicLine {
    LineProfileType type;
    /// upper level
    int j;
    /// lower level
    int i;
    /// oscillator strength
    T f;
    /// natural broadening
    T g_natural;
    /// Einstein A (spontaneous emission)
    T Aji;
    /// Einstein B (stimulated emission); frequency
    T Bji;
    /// Einstein B (stimulated emission); wavelength
    T Bji_wavelength;
    /// Einstein B (absorption); frequency
    T Bij;
    /// Einstein B (absorption); wavelength
    T Bij_wavelength;
    /// Rest wavelength [nm]
    T lambda0;
    /// Broadening terms
    std::vector<ScaledExponentsBroadening<T>> broadening;
    /// Specified wavelength grid [nm]
    std::vector<T> wavelength;
};

template <typename T=fp_t>
struct AtomicContinuum {
    /// Upper level index
    int j;
    /// Lower level index
    int i;
    /// Specified wavelength grid [nm]
    std::vector<T> wavelength;
    /// Specified cross-sections [m2]
    std::vector<T> sigma;
};

enum class CollRateType {
    /// Seaton's collision strength
    Omega,
    /// Collisional ionisation by electrons
    CI,
    /// Collisional excitation of neutrals by electrons
    CE,
    /// Collisional excitation by protons
    CP,
    /// Collisional excitation by neutral H
    CH,
    /// Charge exchange with neutral H (downward only)
    ChargeExcH,
    /// Charge exchange with protons (upward only)
    ChargeExcP,
};

template <typename T=fp_t>
struct InterpCollRate {
    /// Upper level index
    int j;
    /// Lower level index
    int i;
    CollRateType type;
    /// Specified temperature grid
    std::vector<T> temperature;
    /// Specified rates at each temperature point
    std::vector<T> data;
};

enum class AtomicTreatment {
    Detailed,
    Golding,
    Active
};

template <typename T=fp_t>
struct ModelAtom {
    Element<T> element;
    std::vector<AtomicLevel<T>> levels;
    std::vector<AtomicLine<T>> lines;
    std::vector<AtomicContinuum<T>> continua;
    std::vector<InterpCollRate<T>> coll_rates;
    // TODO(cmo): Only here temporarily
    AtomicTreatment treatment = AtomicTreatment::Active;

    /// Compute the wavelength [nm] (lambda0 for lines, lambda_edge for
    /// continua) for a transition between levels j and i (j > i)
    T transition_wavelength(int j, int i) {
        using namespace ConstantsF64;
        T delta_e = levels[j].energy - levels[i].energy;
        return T(hc_eV_nm) / delta_e;
    }
};

template <typename T=fp_t>
struct CompLine {
    /// Short wavelength index
    int red_idx;
    /// Long wavelength index
    int blue_idx;

    /// atom index
    int atom = 0;
    /// upper level
    int j;
    /// lower level
    int i;
    LineProfileType type;
    /// oscillator strength
    T f;
    /// natural broadening
    T g_natural;
    /// Einstein A (spontaneous emission)
    T Aji;
    /// Einstein B (stimulated emission); (nm m2) / kJ
    T Bji;
    /// Einstein B (absorption); (nm m2) / kJ
    T Bij;
    /// Rest wavelength [nm]
    T lambda0;
    /// Broadening start index
    int broad_start;
    /// Broadening end index (exclusive)
    int broad_end;

    YAKL_INLINE bool is_active(int la) const {
        return la >= blue_idx && la < red_idx;
    }
};

template <typename T=fp_t>
struct CompCont {
    /// Short wavelength index
    int red_idx;
    /// Long wavelength index
    int blue_idx;

    /// atom index
    int atom = 0;
    /// upper level
    int j;
    /// lower level
    int i;

    /// Cross-section start index
    int sigma_start;
    /// Cross-section end index
    int sigma_end;

    YAKL_INLINE bool is_active(int la) const {
        return la >= blue_idx && la < red_idx;
    }
};

template <typename T=fp_t>
struct CompColl {
    /// atom index
    int atom = 0;
    /// Upper level index
    int j;
    /// Lower level index
    int i;
    /// Collision type
    CollRateType type;
    /// Temperature/rate start index
    int start_idx;
    /// Temperature/rate end index
    int end_idx;
};


inline bool has_gamma(AtomicTreatment t) {
    return (
        (t == AtomicTreatment::Golding)
        || (t == AtomicTreatment::Active)
    );
}

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
struct CompAtom {
    T mass;
    T abundance;
    int Z;
    AtomicTreatment treatment = AtomicTreatment::Active;

    KView<const T*, mem_space> energy;
    KView<const T*, mem_space> g;
    KView<const T*, mem_space> stage;

    KView<const CompLine<T>*, mem_space> lines;
    /// Shared array of broadeners
    KView<const ScaledExponentsBroadening<T>*, mem_space> broadening;

    KView<const T*, mem_space> wavelength;
    KView<const CompCont<T>*, mem_space> continua;
    KView<const T*, mem_space> sigma;

    KView<const CompColl<T>*, mem_space> collisions;
    KView<const T*, mem_space> temperature;
    KView<const T*, mem_space> coll_rates;
};

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
struct LteTerms {
    T mass;
    T abundance;
    KView<const T*, mem_space> energy;
    KView<const T*, mem_space> g;
    KView<const T*, mem_space> stage;
};


struct TransitionIndex {
    int atom;
    u16 kr;
    bool line;
};


template <typename T=fp_t, typename mem_space=DefaultMemSpace>
struct AtomicData {
    KView<const AtomicTreatment*, mem_space> treatment; // num_atom
    KView<const T*, mem_space> mass; // num_atom
    KView<const T*, mem_space> abundance; // num_atom
    KView<const int*, mem_space> Z; // num_atom

    KView<const int*, mem_space> level_start; // num_atom
    KView<const int*, mem_space> num_level; // num_atom
    KView<const int*, mem_space> line_start; // num_atom
    KView<const int*, mem_space> num_line; // num_atom
    KView<const int*, mem_space> cont_start; // num_atom
    KView<const int*, mem_space> num_cont; // num_atom
    KView<const int*, mem_space> coll_start; // num_atom
    KView<const int*, mem_space> num_coll; // num_atom

    KView<const T*, mem_space> energy; // num_atom * num_level
    KView<const T*, mem_space> g; // num_atom * num_level
    KView<const T*, mem_space> stage; // num_atom * num_level

    KView<const CompLine<T>*, mem_space> lines; // num_atom * num_line
    /// Shared array of broadeners
    KView<const ScaledExponentsBroadening<T>*, mem_space> broadening;

    KView<const CompCont<T>*, mem_space> continua; // num_atom * num_cont
    KView<const T*, mem_space> sigma;

    KView<const T*, mem_space> wavelength;
    KView<const TransitionIndex*, mem_space> governing_trans;

    KView<const CompColl<T>*, mem_space> collisions; // num_atom * num_coll
    KView<const T*, mem_space> temperature;
    KView<const T*, mem_space> coll_rates;

    // Active set stuff
    KView<const u16*, mem_space> active_lines;
    KView<const i32*, mem_space> active_lines_start;
    KView<const i32*, mem_space> active_lines_end;
    KView<const u16*, mem_space> active_cont;
    KView<const i32*, mem_space> active_cont_start;
    KView<const i32*, mem_space> active_cont_end;
};

struct MipmapComputeState {
    i32 max_mip_factor;
    i32 la_start;
    i32 la_end;
    const KView<i32*, DefaultMemSpace>& mippable_entries;
    const Fp2d& emis;
    const Fp2d& opac;
    const Fp1d& vx;
    const Fp1d& vy;
    const Fp1d& vz;
};

struct MipmapSubsetState {
    i32 max_mip_factor;
    i32 la_start;
    i32 la_end;
    const Fp2d& emis;
    const Fp2d& opac;
    const Fp1d& vx;
    const Fp1d& vy;
    const Fp1d& vz;
};

struct MipmapTolerance {
    fp_t opacity_threshold;
    fp_t log_chi_mip_variance;
    fp_t log_eta_mip_variance;
};

#else
#endif
