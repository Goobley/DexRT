#if !defined(DEXRT_TYPES_HPP)
#define DEXRT_TYPES_HPP
#include "Config.hpp"
#include "Constants.hpp"
#include "LoopUtils.hpp"

constexpr auto memDevice = yakl::memDevice;
// constexpr auto memDevice = yakl::memHost;
typedef yakl::Array<fp_t, 1, memDevice> Fp1d;
typedef yakl::Array<fp_t, 2, memDevice> Fp2d;
typedef yakl::Array<fp_t, 3, memDevice> Fp3d;
typedef yakl::Array<fp_t, 4, memDevice> Fp4d;
typedef yakl::Array<fp_t, 5, memDevice> Fp5d;

typedef yakl::Array<fp_t const, 1, memDevice> FpConst1d;
typedef yakl::Array<fp_t const, 2, memDevice> FpConst2d;
typedef yakl::Array<fp_t const, 3, memDevice> FpConst3d;
typedef yakl::Array<fp_t const, 4, memDevice> FpConst4d;
typedef yakl::Array<fp_t const, 5, memDevice> FpConst5d;

typedef yakl::Array<fp_t, 1, yakl::memHost> Fp1dHost;
typedef yakl::Array<fp_t, 2, yakl::memHost> Fp2dHost;
typedef yakl::Array<fp_t, 3, yakl::memHost> Fp3dHost;
typedef yakl::Array<fp_t, 4, yakl::memHost> Fp4dHost;
typedef yakl::Array<fp_t, 5, yakl::memHost> Fp5dHost;

typedef yakl::Array<fp_t const, 1, yakl::memHost> FpConst1dHost;
typedef yakl::Array<fp_t const, 2, yakl::memHost> FpConst2dHost;
typedef yakl::Array<fp_t const, 3, yakl::memHost> FpConst3dHost;
typedef yakl::Array<fp_t const, 4, yakl::memHost> FpConst4dHost;
typedef yakl::Array<fp_t const, 5, yakl::memHost> FpConst5dHost;

typedef yakl::SArray<fp_t, 1, 2> vec2;
typedef yakl::SArray<fp_t, 1, 3> vec3;
typedef yakl::SArray<fp_t, 1, 4> vec4;
typedef yakl::SArray<fp_t, 2, 2, 2> mat2x2;
typedef yakl::SArray<int32_t, 2, 2, 2> imat2x2;
typedef yakl::SArray<int32_t, 1, 2> ivec2;
typedef yakl::SArray<int32_t, 1, 3> ivec3;
typedef yakl::SArray<int32_t, 1, 4> ivec4;

typedef Kokkos::LayoutRight Layout;
template <class T, typename... Args>
using KView = Kokkos::View<T, Layout, Args...>;

typedef Kokkos::DefaultExecutionSpace::memory_space DefaultMemSpace;
typedef Kokkos::HostSpace HostSpace;
constexpr bool HostDevSameSpace = std::is_same_v<DefaultMemSpace, HostSpace>;

typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> TeamPolicy;
typedef TeamPolicy::member_type KTeam;
using Kokkos::DefaultExecutionSpace;
typedef DefaultExecutionSpace::scratch_memory_space ScratchSpace;

template <class T>
using ScratchView = KView<T, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

struct Coord2 {
    i32 x;
    i32 z;

    YAKL_INLINE bool operator==(const Coord2& other) const {
        return (x == other.x) && (z == other.z);
    }
};

using yakl::Dims;
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
    yakl::Array<i32, 2, yakl::memDevice> active_probes; // [n, 2 (u, v)]

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
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> active_probes; // [n, 2 (u, v)]

    /// Setup object, low-level
    void init(
        const CascadeStorage& c0,
        bool sparse,
        std::vector<yakl::Array<i32, 2, yakl::memDevice>> active_probes = decltype(active_probes)()
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
    yakl::Array<T, 1, yakl::memDevice> temperature;
    yakl::Array<T, 1, yakl::memDevice> pressure;
    yakl::Array<T, 1, yakl::memDevice> ne;
    yakl::Array<T, 1, yakl::memDevice> nh_tot;
    yakl::Array<T, 1, yakl::memDevice> nh0;
    yakl::Array<T, 1, yakl::memDevice> vturb;
    yakl::Array<T, 1, yakl::memDevice> vx;
    yakl::Array<T, 1, yakl::memDevice> vy;
    yakl::Array<T, 1, yakl::memDevice> vz;
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

template <typename T=fp_t, int mem_space=memDevice>
struct CompAtom {
    T mass;
    T abundance;
    int Z;
    AtomicTreatment treatment = AtomicTreatment::Active;

    yakl::Array<T const, 1, mem_space> energy;
    yakl::Array<T const, 1, mem_space> g;
    yakl::Array<T const, 1, mem_space> stage;

    yakl::Array<CompLine<T> const, 1, mem_space> lines;
    /// Shared array of broadeners
    yakl::Array<ScaledExponentsBroadening<T> const, 1, mem_space> broadening;

    yakl::Array<T const, 1, mem_space> wavelength;
    yakl::Array<CompCont<T> const, 1, mem_space> continua;
    yakl::Array<T const, 1, mem_space> sigma;

    yakl::Array<CompColl<T> const, 1, mem_space> collisions;
    yakl::Array<T const, 1, mem_space> temperature;
    yakl::Array<T const, 1, mem_space> coll_rates;
};

template <typename T=fp_t, int mem_space=memDevice>
struct LteTerms {
    T mass;
    T abundance;
    yakl::Array<T const, 1, mem_space> energy;
    yakl::Array<T const, 1, mem_space> g;
    yakl::Array<T const, 1, mem_space> stage;
};


struct TransitionIndex {
    int atom;
    u16 kr;
    bool line;
};


template <typename T=fp_t, int mem_space=memDevice>
struct AtomicData {
    yakl::Array<AtomicTreatment const, 1, mem_space> treatment; // num_atom
    yakl::Array<T const, 1, mem_space> mass; // num_atom
    yakl::Array<T const, 1, mem_space> abundance; // num_atom
    yakl::Array<int const, 1, mem_space> Z; // num_atom

    yakl::Array<int const, 1, mem_space> level_start; // num_atom
    yakl::Array<int const, 1, mem_space> num_level; // num_atom
    yakl::Array<int const, 1, mem_space> line_start; // num_atom
    yakl::Array<int const, 1, mem_space> num_line; // num_atom
    yakl::Array<int const, 1, mem_space> cont_start; // num_atom
    yakl::Array<int const, 1, mem_space> num_cont; // num_atom
    yakl::Array<int const, 1, mem_space> coll_start; // num_atom
    yakl::Array<int const, 1, mem_space> num_coll; // num_atom

    yakl::Array<T const, 1, mem_space> energy; // num_atom * num_level
    yakl::Array<T const, 1, mem_space> g; // num_atom * num_level
    yakl::Array<T const, 1, mem_space> stage; // num_atom * num_level

    yakl::Array<CompLine<T> const, 1, mem_space> lines; // num_atom * num_line
    /// Shared array of broadeners
    yakl::Array<ScaledExponentsBroadening<T> const, 1, mem_space> broadening;

    yakl::Array<CompCont<T> const, 1, mem_space> continua; // num_atom * num_cont
    yakl::Array<T const, 1, mem_space> sigma;

    yakl::Array<T const, 1, mem_space> wavelength;
    yakl::Array<TransitionIndex const, 1, mem_space> governing_trans;

    yakl::Array<CompColl<T> const, 1, mem_space> collisions; // num_atom * num_coll
    yakl::Array<T const, 1, mem_space> temperature;
    yakl::Array<T const, 1, mem_space> coll_rates;

    // Active set stuff
    yakl::Array<u16 const, 1, mem_space> active_lines;
    yakl::Array<i32 const, 1, mem_space> active_lines_start;
    yakl::Array<i32 const, 1, mem_space> active_lines_end;
    yakl::Array<u16 const, 1, mem_space> active_cont;
    yakl::Array<i32 const, 1, mem_space> active_cont_start;
    yakl::Array<i32 const, 1, mem_space> active_cont_end;
};

struct MipmapComputeState {
    i32 max_mip_factor;
    i32 la_start;
    i32 la_end;
    const yakl::Array<i32, 1, yakl::memDevice>& mippable_entries;
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
