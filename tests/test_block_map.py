from dataclasses import dataclass
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

CUTOFF_TEMP = 250e3
BLOCK_SIZE = 32
data = "../build/snow_atmos_steeper_10Mm.nc"


# https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
def encode_morton_2(x: np.int32, z: np.int32):
  return (part_1_by_1(np.uint32(z)) << 1) + part_1_by_1(np.uint32(x))

# // "Insert" a 0 bit after each of the 16 low bits of x
def part_1_by_1(x: np.uint32):
  x &= 0x0000ffff                   # x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x

# Inverse of Part1By1 - "delete" all odd-indexed bits
def compact_1_by_1(x: np.uint32):
  x &= 0x55555555                   # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff  # x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x

def decode_morton_2(code: np.uint32):
    return compact_1_by_1(code >> 0), compact_1_by_1(code >> 1) # x, z

def get_tile(field, x, z):
    return field[z * BLOCK_SIZE : (z+1) * BLOCK_SIZE, x * BLOCK_SIZE : (x+1) * BLOCK_SIZE]

@dataclass
class BBox:
    min: np.ndarray[np.int32]
    max: np.ndarray[np.int32]

class BlockMap:
    def __init__(self, field, fill_value=0.0):
        assert field.shape[0] // BLOCK_SIZE == field.shape[0] / BLOCK_SIZE, "z not multiple of block size"
        assert field.shape[1] // BLOCK_SIZE == field.shape[1] / BLOCK_SIZE, "x not multiple of block size"

        self.num_x_tiles = int(field.shape[1] // BLOCK_SIZE)
        self.num_z_tiles = int(field.shape[0] // BLOCK_SIZE)
        self.num_tiles = self.num_x_tiles * self.num_z_tiles
        self.fill_value = fill_value

        self.entries = np.ones((self.num_z_tiles, self.num_x_tiles), np.int32) * -1
        self.grids = []

        self.morton_traversal_order = []
        for z in range(self.num_z_tiles):
            for x in range(self.num_x_tiles):
               self.morton_traversal_order.append(encode_morton_2(x, z))
        self.morton_traversal_order = sorted(self.morton_traversal_order)
        self.fill_grids(field)
        self.bbox = BBox(np.array([0, 0], dtype=np.int32), np.array(field.shape, dtype=np.int32))

    def fill_grids(self, field):
        grid_index = 0
        for t in self.morton_traversal_order:
           x, z = decode_morton_2(np.uint32(t))
           tile = get_tile(field, x, z)
           if np.all(tile > CUTOFF_TEMP):
              continue

           self.grids.append(np.ascontiguousarray(tile))
           self.entries[z, x] = grid_index
           grid_index += 1

    def order(self):
        tiles = np.zeros((self.num_tiles, 2))
        for i, t in enumerate(self.morton_traversal_order):
           tiles[i, :] = decode_morton_2(t)
        return tiles

class Accessor:
    def __init__(self, block_map: BlockMap):
        self.block_map = block_map
        self.tile_key = None
        self.tile = None

    def get(self, x, z):
        tile_x = int(x // BLOCK_SIZE)
        inner_x = x % BLOCK_SIZE
        tile_z = int(z // BLOCK_SIZE)
        inner_z = z % BLOCK_SIZE
        if self.tile_key == (tile_x, tile_z):
            return self.tile[inner_z, inner_x]

        tile_index = self.block_map.entries[tile_z, tile_x]
        if tile_index == -1:
            # raise ValueError("Accesing invalid tile")
            return self.block_map.fill_value()

        self.tile = self.block_map.grids[tile_index]
        self.tile_key = (tile_x, tile_z)
        return self.tile[inner_z, inner_x]

    def has_leaves(self, x, z):
        tile_x = int(x // BLOCK_SIZE)
        tile_z = int(z // BLOCK_SIZE)
        return self.block_map.entries[tile_z, tile_x] != -1


@dataclass
class Ray:
    o: np.ndarray
    d: np.ndarray
    t0: float
    t1: float

    def at(self, t):
        return self.o + t * self.d

    # nanovdb ray style
    def intersects(self, bbox: BBox):
        t0, t1 = self.t0, self.t1
        for i in range(2):
            a = float(bbox.min[i])
            b = float(bbox.max[i] + 1)
            if a >= b:
                return False, (t0, t1)

            a = (a - self.o[i]) / self.d[i]
            b = (b - self.o[i]) / self.d[i]
            if a > b:
                b, a = a, b
            if a > t0:
                t0 = a
            if b < t1:
                t1 = b
            if t0 > t1:
                return False, (t0, t1)
        return True, (t0, t1)

    def clip(self, bbox: BBox):
        hit, (t0, t1) = self.intersects(bbox)
        if hit:
            # NOTE(cmo): Can add start_clipped here if self.t0 != t0
            self.t0 = t0
            self.t1 = t1
        return hit # whether the clipped ray is inside the bbox

def round_down(coord):
    return np.floor(coord).astype(np.int32)

class TwoLevelDDA:
    def __init__(self, accessor: Accessor, ray: Ray):
        self.accessor = accessor
        self.ray = ray

        if not ray.clip(self.accessor.block_map.bbox):
            print("Can't trace -- need to signal this")

        end_pos = ray.at(ray.t1)
        self.start_pos = ray.at(ray.t0)
        self.curr_coord = round_down(self.start_pos)
        self.final_coord = round_down(end_pos)
        self.next_coord = np.copy(self.curr_coord)

        self.step_size = 1
        if not accessor.has_leaves(*self.curr_coord):
            self.step_size = BLOCK_SIZE

        self.step = np.array([0, 0], dtype=np.int32)
        self.next_hit = np.array([0.0, 0.0])
        self.delta = np.array([0.0, 0.0])

        # NOTE(cmo): Assume d is normalised correctly
        inv_dir = 1.0 / ray.d
        for axis in range(2):
            if ray.d[axis] == 0:
                self.step[axis] = 0
                self.next_hit[axis] = 1e24
            elif inv_dir[axis] > 0:
                self.step[axis] = 1
                self.next_hit[axis] = (self.curr_coord[axis] + self.step_size - self.start_pos[axis]) * inv_dir[axis]
                self.delta[axis] = inv_dir[axis]
            else:
                self.step[axis] = -1
                # NOTE(cmo): This is probably wrong for an arbitrary input to a base cell
                self.next_hit[axis] = (self.curr_coord[axis] - self.start_pos[axis]) * inv_dir[axis]
                self.delta[axis] = -inv_dir[axis]

        self.t = self.ray.t0
        self.dt = 0.0

    def next_intersection(self):
        self.curr_coord[:] = self.next_coord

        axis = 0
        if self.next_hit[1] < self.next_hit[0]:
            axis = 1

        prev_t = self.t
        new_t = self.next_hit[axis]
        self.next_hit[axis] += self.step_size * self.delta[axis]
        self.next_coord[axis] += self.step_size * self.step[axis]

        if new_t >= self.ray.t1 and prev_t < self.ray.t1:
            prev_hit = self.ray.at(prev_t)
            end_pos = self.ray.at(self.ray.t1)
            # NOTE(cmo): This seems superfluous vs self.ray.t1 - prev_t
            self.dt = np.sqrt(np.sum((end_pos - prev_hit)**2))
            new_t = self.ray.t1
        else:
            self.dt = new_t - prev_t

        self.t = new_t
        return new_t <= self.ray.t1

if __name__ == "__main__":
    field = xr.open_dataset(data).temperature.values

    block_map = BlockMap(field)
    order = block_map.order() * BLOCK_SIZE + BLOCK_SIZE / 2
    plt.imshow(field, origin="lower")
    plt.plot(order[:, 0], order[:, 1])
    for x in range(BLOCK_SIZE, field.shape[1], BLOCK_SIZE):
        plt.axvline(x + 0.5, c='k', lw=0.5)
    for z in range(BLOCK_SIZE, field.shape[0], BLOCK_SIZE):
        plt.axhline(z + 0.5, c='k', lw=0.5)

    acc = Accessor(block_map)
    ray = Ray(o=np.array([1, 1]), d=np.array([0.7, np.sqrt(1.0 - 0.7**2)]), t0=0.0, t1=1e4)
    hdda = TwoLevelDDA(acc, ray)

    ts = [hdda.t]
    while hdda.next_intersection():
        ts.append(hdda.t)

    ts = np.array(ts)
    hits = ray.at(ts[:, None])
    plt.plot(hits[:, 0], hits[:, 1], 'x')
    # NOTE(cmo): Weird offsets going on here. Start by making image sane -- centre pixels on 0.5s



