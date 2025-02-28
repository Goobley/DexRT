
#if !defined(DEXRT_MORTON_CODES_HPP)
#define DEXRT_MORTON_CODES_HPP
#include "Config.hpp"
#include "Types.hpp"
// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// "Insert" a 0 bit after each of the 16 low bits of x
YAKL_INLINE uint32_t part_1_by_1(uint32_t x)
{
    x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
YAKL_INLINE uint32_t part_1_by_2(uint32_t x)
{
  x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x;
}

// Inverse of Part1By1 - "delete" all odd-indexed bits
YAKL_INLINE uint32_t compact_1_by_1(uint32_t x)
{
    x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
YAKL_INLINE uint32_t compact_1_by_2(uint32_t x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}

template <int N>
YAKL_INLINE uint32_t encode_morton(const Coord<N>& coord);

template <>
YAKL_INLINE uint32_t encode_morton<2>(const Coord2& coord) {
    return (part_1_by_1(uint32_t(coord.z)) << 1) + part_1_by_1(uint32_t(coord.x));
}

template <>
YAKL_INLINE uint32_t encode_morton<3>(const Coord3& coord) {
    return (
        (part_1_by_2(uint32_t(coord.z)) << 2)
        + (part_1_by_2(uint32_t(coord.y)) << 1)
        + part_1_by_2(uint32_t(coord.x))
    );
}

template <int N>
YAKL_INLINE Coord<N> decode_morton(uint32_t code);

template <>
YAKL_INLINE Coord2 decode_morton<2>(uint32_t code) {
    return Coord2 {
        .x = i32(compact_1_by_1(code)),
        .z = i32(compact_1_by_1(code >> 1))
    };
}

template <>
YAKL_INLINE Coord3 decode_morton<3>(uint32_t code) {
    return Coord3 {
        .x = i32(compact_1_by_2(code)),
        .y = i32(compact_1_by_2(code >> 1)),
        .z = i32(compact_1_by_2(code >> 2))
    };
}

template <int N>
struct MaxMortonValue {
};

template <>
struct MaxMortonValue<2> {
    constexpr static i32 value = (1 << 16) - 1;
};

template <>
struct MaxMortonValue<3> {
    constexpr static i32 value = (1 << 10) - 1;
};

#endif