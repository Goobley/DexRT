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
        self.bbox = BBox(np.array([0, 0], dtype=np.int32), np.array([field.shape[1], field.shape[0]], dtype=np.int32))

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
        if self.tile_key == (tile_z, tile_x):
            return True

        tile_idx = self.block_map.entries[tile_z, tile_x]
        result = tile_idx != -1
        if result:
            self.tile = self.block_map.grids[tile_idx]
            self.tile_key = (tile_x, tile_z)
        return result


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
            b = float(bbox.max[i])
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
    def __init__(self, accessor: Accessor, ray: Ray, refined_size: int=1):
        self.accessor = accessor
        self.ray = ray
        self.refined_size = refined_size

        if not ray.clip(self.accessor.block_map.bbox):
            raise ValueError("Outside domain")

        eps = 1e-6
        end_pos = ray.at(ray.t1 - eps)
        start_pos = ray.at(ray.t0 + eps)
        self.curr_coord = round_down(start_pos)
        self.curr_coord = np.minimum(self.curr_coord, self.accessor.block_map.bbox.max-1)
        self.step_size = refined_size
        if not accessor.has_leaves(*self.curr_coord):
            self.step_size = BLOCK_SIZE
        self.curr_coord = self.curr_coord & (~np.uint32(self.step_size-1))

        self.final_coord = round_down(end_pos)
        self.next_coord = np.copy(self.curr_coord)

        self.step = np.array([0, 0], dtype=np.int32)
        self.next_hit = np.array([0.0, 0.0])
        self.delta = np.array([0.0, 0.0])
        self.t = self.ray.t0

        # NOTE(cmo): Assume d is normalised correctly
        inv_dir = 1.0 / ray.d
        for axis in range(2):
            if ray.d[axis] == 0:
                self.step[axis] = 0
                self.next_hit[axis] = 1e24
            elif inv_dir[axis] > 0:
                self.step[axis] = 1
                self.next_hit[axis] = self.t + (self.curr_coord[axis] + self.step_size - start_pos[axis]) * inv_dir[axis]
                self.delta[axis] = inv_dir[axis]
            else:
                self.step[axis] = -1
                self.next_hit[axis] = self.t + (self.curr_coord[axis] - start_pos[axis]) * inv_dir[axis]
                self.delta[axis] = -inv_dir[axis]

        self.compute_axis_and_dt()

    def compute_axis_and_dt(self):
        self.step_axis = np.argmin(self.next_hit)
        next_t = self.next_hit[self.step_axis]

        if next_t <= self.t:
            print(f"{self.next_hit[self.step_axis]} <= {self.t} clamping ({self.curr_coord})")
            self.next_hit[self.step_axis] += self.t - 0.999999 * self.next_hit[self.step_axis] + 1.0e-6
            next_t = self.next_hit[self.step_axis]

        if next_t > self.ray.t1:
            self.dt = self.ray.t1 - self.t
        else:
            self.dt = self.next_hit[self.step_axis] - self.t
        self.dt = max(self.dt, 0.0)

    def update_step_size(self, step_size):
        if step_size == self.step_size:
            return

        self.step_size = step_size
        curr_pos = ray.at(self.t)
        self.curr_coord = round_down(curr_pos) & (~np.uint32(self.step_size-1))
        print(f"pos {curr_pos}, clamping to {self.curr_coord}")
        inv_dir = 1.0 / ray.d
        for axis in range(2):
            if self.step[axis] == 0:
                continue

            self.next_hit[axis] = self.t + (self.curr_coord[axis] - curr_pos[axis]) * inv_dir[axis]
            if self.step[axis] > 0:
                self.next_hit[axis] += self.step_size * inv_dir[axis]

        self.compute_axis_and_dt()

    def exhausted(self):
        return self.t >= self.ray.t1

    def next_intersection(self):
        axis = self.step_axis

        self.t = self.next_hit[axis]
        self.next_hit[axis] += self.step_size * self.delta[axis]
        self.curr_coord[axis] += self.step_size * self.step[axis]

        self.compute_axis_and_dt()
        return self.t < self.ray.t1

    def step_through(self):
        while self.next_intersection():
            has_leaves = self.accessor.has_leaves(*self.curr_coord)
            # Already marching at size 1 through refined region, return immediately
            if self.step_size == self.refined_size and has_leaves:
                return True

            # Refine, then return first intersection inside refined region
            if has_leaves:
                print("refine", self.curr_coord)
                # NOTE(cmo): A couple of different approaches for catching self-intersection with outer boundary
                BIAS = False
                if BIAS:
                    self.t += 1e-6
                self.update_step_size(self.refined_size)
                if not BIAS:
                    if not self.accessor.has_leaves(*self.curr_coord):
                        continue
                return True

            # Not in refined region, so go back to big steps
            if self.step_size == self.refined_size:
                print("coarsen", self.curr_coord)
                # NOTE(cmo): Boop us away from the boundary of this coarsened cell to avoid getting stuck.
                self.t += 0.01
                self.update_step_size(BLOCK_SIZE)
        return False

    def can_sample(self):
        return self.step_size == self.refined_size


def ray_from_start_end(start, end):
    d = (end - start)
    length = np.linalg.norm(d)
    d /= length

    ray = Ray(o=start, d=d, t0=0.0, t1=length)
    return ray


if __name__ == "__main__":
    field = xr.open_dataset(data).temperature.values
    # field = np.random.uniform(size=(16, 16))

    block_map = BlockMap(field)
    order = block_map.order() * BLOCK_SIZE + BLOCK_SIZE / 2
    x_edges = np.arange(field.shape[1]+1)
    z_edges = np.arange(field.shape[0]+1)
    plt.pcolormesh(x_edges, z_edges, field)
    plt.plot(order[:, 0], order[:, 1])
    for x in range(BLOCK_SIZE, field.shape[1], BLOCK_SIZE):
        plt.axvline(x, c='k', lw=0.5)
    for z in range(BLOCK_SIZE, field.shape[0], BLOCK_SIZE):
        plt.axhline(z, c='k', lw=0.5)

    acc = Accessor(block_map)
    # ray = Ray(o=np.array([1, 1]), d=np.array([0.6, np.sqrt(1.0 - 0.6**2)]), t0=0.0, t1=1e4)
    # ray = ray_from_start_end(np.array([2.2, 3.5]), np.array([3.1, 10.23]))
    # ray = ray_from_start_end(np.array([1.91421, 3.085815]), np.array([0.5, 4.5]))
    # ray = ray_from_start_end(np.array([2.1, 3.1]), np.array([4.8, 5.8]))
    # ray = ray_from_start_end(np.array([2.8, 18.4]), np.array([21.0, 2342.0]))
    # ray = ray_from_start_end(np.array([17.0, 8.5]), np.array([14.1, 5.6]))
    ray = ray_from_start_end(np.array([1.91421, 3.085815]) + 200, np.array([0.5, 4.5]) + 200)
    # ray = ray_from_start_end(np.array([800.0, 247.3]), np.array([564.0, 564.0]))
    # ray = ray_from_start_end()
    hdda = TwoLevelDDA(acc, ray, 1)

    # ts = [hdda.t]
    # while hdda.step_through():
    #     ts.append(hdda.t)
    ts = []
    i = 0
    while not hdda.exhausted() and i < 100:
        if hdda.can_sample():
            ts.append(hdda.t)
            if not acc.has_leaves(*hdda.curr_coord):
                raise ValueError("OOB")
            print(hdda.curr_coord, hdda.t, hdda.dt, ray.at(hdda.t))
        hdda.step_through()
        i += 1

    ts = np.array(ts)
    hits = ray.at(ts[:, None])
    plt.plot(hits[:, 0], hits[:, 1], 'x')



