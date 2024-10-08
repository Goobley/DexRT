
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

struct Coord2 {
    i32 x;
    i32 z;

    YAKL_INLINE bool operator==(const Coord2& other) const {
        return (x == other.x) && (z == other.z);
    }
};

YAKL_INLINE uint32_t encode_morton_2(const Coord2& coord) {
  return (part_1_by_1(uint32_t(coord.z)) << 1) + part_1_by_1(uint32_t(coord.x));
}

YAKL_INLINE Coord2 decode_morton_2(uint32_t code) {
    return Coord2 {
        .x = i32(compact_1_by_1(code)),
        .z = i32(compact_1_by_1(code >> 1))
    };
}

#endif