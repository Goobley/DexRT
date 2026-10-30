#if !defined(DEXRT_RAD16_HPP)
#define DEXRT_RAD16_HPP

#include "Types.hpp"
#include "LoopUtils.hpp"
#include <bit>
#include <cmath>

/// 16-bit storage for non-negative radiation quantities (e.g. J): no sign,
/// a 6-bit exponent and binary16's 10-bit mantissa (relative precision 2^-11).
/// Normal values span [2^-39, 2^24) (subnormals down to 2^-49), wide enough for
/// intensities without per-wavelength scales. Rounds to nearest even;
/// saturates at the largest value; negative or NaN inputs store 0.

constexpr u16 RAD16_MAX_BITS = 0xffff;
/// f32 biased exponent of the format's exponent 1 (2^-39).
constexpr u32 RAD16_EXP_OFFSET = 87;

YAKL_INLINE f32 rad16_to_float(u16 h) {
    const u32 exp = h >> 10;
    const u32 mant = h & 0x3ff;
    if (exp == 0) {
        // NOTE(cmo): Zero and subnormals: mant * 2^-49.
        return f32(mant) * 1.7763568394002505e-15f;
    }
    return std::bit_cast<f32>(((exp + RAD16_EXP_OFFSET) << 23) | (mant << 13));
}

YAKL_INLINE u16 float_to_rad16(f32 f) {
    if (!(f > 0.0f)) {
        return 0;
    }
    u32 x = std::bit_cast<u32>(f);
    // NOTE(cmo): Below the smallest normal (2^-39): mant = round(f 2^49).
    if (x < ((RAD16_EXP_OFFSET + 1) << 23)) {
        return u16(std::nearbyint(f * 562949953421312.0f));
    }
    x = x - (RAD16_EXP_OFFSET << 23) + 0xfffu + ((x >> 13) & 1u);
    const u32 h = x >> 13;
    return h > RAD16_MAX_BITS ? RAD16_MAX_BITS : u16(h);
}

/// A [row, cell] radiation array, 16-bit (float_to_rad16) if
/// HALF_RADIATION_STORAGE, else fp_t. Accumulate in fp_t and store once.
template <int Mem>
struct Rad16ArrayT {
    using Storage = std::conditional_t<HALF_RADIATION_STORAGE, u16, fp_t>;
    yakl::Array<Storage, 2, Mem> data;

    Rad16ArrayT() = default;
    Rad16ArrayT(const char* label, i64 rows, i64 cells) : data(label, rows, cells) {}

    YAKL_INLINE static Storage encode(fp_t v) {
        if constexpr (HALF_RADIATION_STORAGE) {
            return float_to_rad16(f32(v));
        } else {
            return v;
        }
    }
    YAKL_INLINE static fp_t decode(Storage v) {
        if constexpr (HALF_RADIATION_STORAGE) {
            return fp_t(rad16_to_float(v));
        } else {
            return v;
        }
    }
    YAKL_INLINE fp_t operator()(i64 row, i64 k) const {
        return decode(data(row, k));
    }
    YAKL_INLINE void set(i64 row, i64 k, fp_t v) const {
        data(row, k) = encode(v);
    }
    YAKL_INLINE void add(i64 row, i64 k, fp_t v) const {
        set(row, k, (*this)(row, k) + v);
    }
    void fill(fp_t v) const {
        data = encode(v);
    }
    bool initialized() const {
        return data.initialized();
    }
    i64 extent(int i) const {
        return data.extent(i);
    }
    /// An unowned view of rows [row_start, row_start + num_rows).
    yakl::Array<Storage, 2, Mem> rows(i64 row_start, i64 num_rows) const {
        return yakl::Array<Storage, 2, Mem>("rad16_rows", data.data() + row_start * extent(1), num_rows, extent(1));
    }
    /// Store rows [0, num_rows) of src as rows [row_start, row_start + num_rows).
    void store_rows(i64 row_start, const Fp2d& src, i64 num_rows) const requires (Mem == yakl::memDevice) {
        const Rad16ArrayT self = *this;
        dex_parallel_for("Radiation store rows", FlatLoop<2>(num_rows, extent(1)), YAKL_LAMBDA(i64 r, i64 k) {
            self.set(row_start + r, k, src(r, k));
        });
        Kokkos::fence();
    }
    /// An fp_t copy.
    Fp2d to_fp() const requires (Mem == yakl::memDevice) {
        Fp2d out("rad16_fp", extent(0), extent(1));
        const Rad16ArrayT self = *this;
        dex_parallel_for("Radiation to fp", FlatLoop<2>(extent(0), extent(1)), YAKL_LAMBDA(i64 r, i64 k) {
            out(r, k) = self(r, k);
        });
        Kokkos::fence();
        return out;
    }
    static Rad16ArrayT from_fp(const Fp2d& a) requires (Mem == yakl::memDevice) {
        Rad16ArrayT out("rad16", a.extent(0), a.extent(1));
        out.store_rows(0, a, a.extent(0));
        return out;
    }
    /// Rows [row_start, row_start + num_rows) as fp_t.
    Fp2dHost decode_rows(i64 row_start, i64 num_rows) const requires (Mem == yakl::memHost) {
        Fp2dHost out("rad16_decoded", num_rows, extent(1));
        for (i64 r = 0; r < num_rows; ++r) {
            for (i64 k = 0; k < extent(1); ++k) {
                out(r, k) = (*this)(row_start + r, k);
            }
        }
        return out;
    }
};
typedef Rad16ArrayT<yakl::memDevice> Rad16Array;
typedef Rad16ArrayT<yakl::memHost> Rad16ArrayHost;

/// Copy rows [0, num_rows) of src into rows [row_start, row_start + num_rows)
/// of host, encoding on the device in stage first if 16-bit.
template <typename HostArray>
void page_rows_to_host(const HostArray& host, i64 row_start, const Fp2d& src, i64 num_rows, const Rad16Array& stage) {
    const auto dst = host.rows(row_start, num_rows);
    if constexpr (HALF_RADIATION_STORAGE) {
        stage.store_rows(0, src, num_rows);
        stage.rows(0, num_rows).deep_copy_to(dst);
    } else {
        Fp2d("rows", src.data(), num_rows, src.extent(1)).deep_copy_to(dst);
    }
    Kokkos::fence();
}

#else
#endif
