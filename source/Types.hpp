#if !defined(DEXRT_TYPES_HPP)
#define DEXRT_TYPES_HPP
#include "Config.hpp"
#include "Constants.hpp"

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
typedef yakl::SArray<int, 2, 2, 2> imat2x2;
typedef yakl::SArray<int32_t, 1, 2> ivec2;

using yakl::c::parallel_for;
using yakl::c::SimpleBounds;
using yakl::Dims;

struct CascadeDims {
    ivec2 num_probes;
    int num_flat_dirs;
    int wave_batch;
    int num_incl;
};

struct InclQuadrature {
    Fp1d muy;
    Fp1d wmuy;
};

struct RadianceInterval {
    fp_t I = FP(0.0);
    fp_t tau = FP(0.0);
};

struct RayProps {
    vec2 start;
    vec2 end;
    vec2 dir;
    vec2 centre;
};

struct CascadeState {
    int num_cascades;
    std::vector<Fp1d> i_cascades;
    std::vector<Fp1d> tau_cascades;
    Fp3d eta; // [z, x, wave]
    Fp3d chi; // [z, x, wave]
};

struct DeviceCascadeState {
    int num_cascades;
    int n;
    Fp1d cascade_I;
    Fp1d cascade_tau;
    FpConst1d upper_I;
    FpConst1d upper_tau;
    Fp3d eta; // [z, x, wave]
    Fp3d chi; // [z, x, wave]
};

template <typename Bc>
struct CascadeStateAndBc {
    DeviceCascadeState state;
    Bc bc;
};

struct Atmosphere {
    fp_t voxel_scale;
    fp_t offset_x = FP(0.0);
    fp_t offset_y = FP(0.0);
    fp_t offset_z = FP(0.0);
    Fp2d temperature;
    Fp2d pressure;
    Fp2d ne;
    Fp2d nh_tot;
    Fp2d vturb;
    Fp2d vx;
    Fp2d vy;
    Fp2d vz;
};

template <typename T=fp_t>
struct FlatAtmosphere {
    fp_t voxel_scale;
    yakl::Array<T, 1, yakl::memDevice> temperature;
    yakl::Array<T, 1, yakl::memDevice> pressure;
    yakl::Array<T, 1, yakl::memDevice> ne;
    yakl::Array<T, 1, yakl::memDevice> nh_tot;
    yakl::Array<T, 1, yakl::memDevice> vturb;
    yakl::Array<T, 1, yakl::memDevice> vx;
    yakl::Array<T, 1, yakl::memDevice> vy;
    yakl::Array<T, 1, yakl::memDevice> vz;
};

struct MipmapState {
    Fp3d emission;
    Fp3d absorption;
    std::vector<Fp3d> emission_mipmaps;
    std::vector<Fp3d> absorption_mipmaps;
    yakl::SArray<int, 1, MAX_CASCADE+1> cumulative_mipmap_factor;
};

struct CascadeRTState {
    int mipmap_factor;
    FpConst3d eta;
    FpConst3d chi;
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

template <typename T=fp_t>
struct ModelAtom {
    Element<T> element;
    std::vector<AtomicLevel<T>> levels;
    std::vector<AtomicLine<T>> lines;
    std::vector<AtomicContinuum<T>> continua;
    std::vector<InterpCollRate<T>> coll_rates;

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

template <typename T=fp_t, int mem_space=memDevice>
struct CompAtom {
    T mass;
    T abundance;
    int Z;

    yakl::Array<T const, 1, mem_space> energy;
    yakl::Array<T const, 1, mem_space> g;
    yakl::Array<T const, 1, mem_space> stage;

    yakl::Array<CompLine<T> const, 1, mem_space> lines;
    /// Shared array of broadeners
    yakl::Array<ScaledExponentsBroadening<T> const, 1, mem_space> broadening;

    /// Temporary per atom wavelength grid
    yakl::Array<T const, 1, mem_space> wavelength;
    yakl::Array<CompCont<T> const, 1, mem_space> continua;
    yakl::Array<T const, 1, mem_space> sigma;

    yakl::Array<CompColl<T> const, 1, mem_space> collisions;
    yakl::Array<T const, 1, mem_space> temperature;
    yakl::Array<T const, 1, mem_space> coll_rates;
};

#else
#endif
