import copy
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import astropy.units as u
import lightweaver as lw
from lightweaver.rh_atoms import CaII_atom, H_6_atom
import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from weno4 import weno4
from scipy.ndimage import gaussian_filter1d

def collisionless_CaII():
    ca = CaII_atom()
    ca.collisions = []
    lw.reconfigure_atom(ca)
    return ca

def compute_intersection_t(p0, d):
    # NOTE(cmo): Units of Rs
    a = 1.0
    b = 2.0 * (p0 @ d)
    c = p0 @ p0 - 1.0
    delta = b*b - 4.0 * a * c

    if delta < 0:
        return None
    return (-b - np.sqrt(delta)) / (2.0 * a)

def compute_intersection_length(p0, d):
    # NOTE(cmo): Units of Rs
    a = 1.0
    b = 2.0 * (p0 @ d)
    c = p0 @ p0 - 1.0
    delta = b*b - 4.0 * a * c

    if delta < 0:
        return None
    t1 = (-b - np.sqrt(delta)) / (2.0 * a)
    t2 = (-b + np.sqrt(delta)) / (2.0 * a)
    return t2 - t1

Rs = np.float64(const.R_sun.value)
def compute_intersection_angle(p0, d):
    # NOTE(cmo): Units of Rs
    p0 = p0.copy()
    p0 /= Rs
    p0[2] += 1.0
    t_min = compute_intersection_t(p0, d)
    if t_min is None:
        return None
    int_loc = p0 + t_min * d
    # cosphi = -(d @ int_loc) / Rs
    # NOTE(cmo): Rs is 1.0
    cosphi = -(d @ int_loc)
    return cosphi

if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    dex_J = np.array(ds["image"][...])
    dex_wave = np.array(ds["wavelength"][...])
    dex_eta = np.array(ds["eta"][...])
    dex_chi = np.array(ds["chi"][...])
    dex_pops = np.array(ds["pops"][...])

    ds = netCDF4.Dataset("build/atmos.nc", "r")
    nz = ds.dimensions["z"].size
    z_grid = np.ascontiguousarray((np.arange(nz, dtype=np.float64) * ds["voxel_scale"][...])[::-1])
    sample_idx = ds["ne"].shape[1] // 2
    temperature = np.ascontiguousarray(ds["temperature"][:, sample_idx][::-1]).astype(np.float64)
    ne = np.ascontiguousarray(ds["ne"][:, sample_idx][::-1]).astype(np.float64)
    nhtot = np.ascontiguousarray(ds["nh_tot"][:, sample_idx][::-1]).astype(np.float64)
    vturb = np.ascontiguousarray(ds["vturb"][:, sample_idx][::-1]).astype(np.float64)

    voxel_scale = np.float32(ds["voxel_scale"][...])
    offset_x = np.float32(ds["offset_x"][...])
    offset_z = np.float32(ds["offset_z"][...])

    bc_ctx = pw.compute_falc_bc_ctx(["H", "Ca"])
    # bc_table = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.1, 1.0, 10))
    orig_bc_table = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20))
    # bc_table = pw.tabulate_bc(bc_ctx)
    I_with_zero = np.zeros((orig_bc_table["I"].shape[0], orig_bc_table["I"].shape[1] + 1))
    I_with_zero[:, 1:] = orig_bc_table["I"][...]
    orig_bc_table["I"] = I_with_zero
    orig_bc_table["mu_grid"] = np.concatenate([[0], orig_bc_table["mu_grid"]])

    bc_table = copy.copy(orig_bc_table)
    # NOTE(cmo): Filter table
    # bc_table["mu_grid"] = np.linspace(0.0, 1.0, 21)
    # bc_table["I"] = np.zeros((bc_table["wavelength"].shape[0], bc_table["mu_grid"].shape[0]))
    # cone_half_angular_width = 2.0 * np.pi / 1024
    # gl_sample = np.sqrt(3 / 5)
    # mu_weights = [2.5 / 9, 4 / 9, 2.5 / 9]
    # for mu_idx, mu in enumerate(bc_table["mu_grid"]):
    #     theta = np.arccos(mu)
    #     thetam = theta - gl_sample * cone_half_angular_width
    #     thetap = theta + gl_sample * cone_half_angular_width
    #     mum = np.cos(thetam)
    #     mup = np.cos(thetap)

    #     frac_idx = np.interp(mu, orig_bc_table["mu_grid"], np.arange(orig_bc_table["mu_grid"].shape[0]))
    #     int_idx = int(frac_idx)
    #     idxp = min(int_idx + 1, orig_bc_table["mu_grid"].shape[0] - 1)
    #     t = frac_idx - int_idx
    #     sample = (1 - t) * orig_bc_table["I"][:, int_idx] + t * orig_bc_table["I"][:, idxp]
    #     bc_table["I"][:, mu_idx] += mu_weights[1] * sample

    #     frac_idx = np.interp(mum, orig_bc_table["mu_grid"], np.arange(orig_bc_table["mu_grid"].shape[0]))
    #     int_idx = int(frac_idx)
    #     t = frac_idx - int_idx
    #     sample = (1 - t) * orig_bc_table["I"][:, int_idx] + t * orig_bc_table["I"][:, idxp]
    #     bc_table["I"][:, mu_idx] += mu_weights[0] * sample

    #     frac_idx = np.interp(mup, orig_bc_table["mu_grid"], np.arange(orig_bc_table["mu_grid"].shape[0]))
    #     int_idx = int(frac_idx)
    #     t = frac_idx - int_idx
    #     sample = (1 - t) * orig_bc_table["I"][:, int_idx] + t * orig_bc_table["I"][:, idxp]
    #     bc_table["I"][:, mu_idx] += mu_weights[0] * sample

    # prev_I = np.copy(bc_table["I"])
    # # bc_table["I"] = gaussian_filter1d(bc_table["I"], sigma=1, mode="nearest")
    # bc_table["I"][:, 0] = 0.0



    bc_provider = pw.TabulatedPromBcProvider(**bc_table)

    fil_atmos = lw.Atmosphere.make_1d(
        lw.ScaleType.Geometric,
        z_grid,
        temperature,
        vlos=np.zeros_like(z_grid),
        vturb=vturb,
        ne=ne,
        nHTot=nhtot,
        lowerBc=pw.ConePromBc("filament", bc_provider, altitude_m=ds["offset_z"][...]),
        upperBc=lw.ZeroRadiation(),
    )
    fil_atmos.quadrature(10)

    # a_set = lw.RadiativeSet([H_6_atom(), collisionless_CaII()])
    a_set = lw.RadiativeSet([H_6_atom(), CaII_atom()])
    a_set.set_active("H", "Ca")
    eq_pops = a_set.compute_eq_pops(fil_atmos)
    # spect = a_set.compute_wavelength_grid(lambdaReference=CaII_atom().lines[-1].lambda0 - 1.0)
    spect = a_set.compute_wavelength_grid()

    # ctx = lw.Context(fil_atmos, spect, eq_pops, formalSolver="piecewise_linear_1d")
    # ctx.background.eta[...] = 0.0
    # ctx.background.chi[...] = 0.0
    # ctx.background.sca[...] = 0.0
    # ctx.depthData.fill = True
    # # ctx.formal_sol_gamma_matrices()
    # lw.iterate_ctx_se(ctx, popsTol=1e-2, JTol=1.0)
    # fil_atmos.zLowerBc.final_synthesis = True
    # Ivert_fil = lw.convert_specific_intensity(
    #     ctx.spect.wavelength,
    #     ctx.compute_rays(mus=1.0),
    #     outUnits="kW / (m2 nm sr)"
    # )
    # plt.ion()
    # plt.plot(ctx.spect.wavelength, Ivert_fil)

    ray_output = netCDF4.Dataset("build/ray_output.nc")
    # central_Ivert_fil = ray_output["I_0"][:, ray_output["I_0"].shape[1]//2]
    dex_wave = ray_output["wavelength"][...]
    # plt.plot(dex_wave, central_Ivert_fil)

    # def compute_ray_idx(i, cone_half_angular_width=None):
    #     probe_I = np.zeros((32, 32))
    #     for probe_u in range(0, 32):
    #         for probe_v in range(0, 32):
    #             probe_x = (probe_u + 0.5) * probe_spacing
    #             probe_z = (probe_v + 0.5) * probe_spacing

    #             # TODO(cmo): Try a 3pt quadrature?
    #             # j_weights = [0.25, 0.5, 0.25]
    #             # j_sample = [-1, 0, 1]
    #             # for j in range(3):
    #             # angle = (i + j_sample[j] + 0.5) * 2.0 * np.pi / 1024
    #             angle = (i + 0.5) * 2.0 * np.pi / 1024
    #             mu_x = np.cos(angle)
    #             mu_z = np.sin(angle)

    #             gl_sample = np.sqrt(3 / 5)
    #             mu = np.array([mu_x, 0.0, mu_z])
    #             if mu_z < 0.0:
    #                 pos = np.array([
    #                     probe_x * voxel_scale + offset_x,
    #                     # 0.0,
    #                     0.0,
    #                     probe_z * voxel_scale + offset_z,
    #                 ])
    #                 if cone_half_angular_width is None:
    #                     mu_hits = [compute_intersection_angle(pos, mu)]
    #                     mu_weights = [1.0]
    #                 else:
    #                     mu_hits = []
    #                     mu_weights = [2.5 / 9, 4 / 9, 2.5 / 9]
    #                     for m in [-1, 0, 1]:
    #                         cone_angle = angle + m * gl_sample * cone_half_angular_width
    #                         cone_mu = np.array([np.cos(cone_angle), 0.0, np.sin(cone_angle)])
    #                         mu_hits.append(compute_intersection_angle(pos, cone_mu))

    #                 wave_idx = np.searchsorted(bc_table["wavelength"], dex_wave[20])

    #                 if any(m is not None for m in mu_hits):
    #                     Ibc = 0.0
    #                     for mu_idx, mu_hit in enumerate(mu_hits):
    #                         if mu_hit is None:
    #                             continue
    #                         # Ibc += mu_weights[mu_idx] * bc_provider.compute_I(dex_wave[20], mu_hit)
    #                         Ibc += mu_weights[mu_idx] * np.interp(mu_hit, bc_table["mu_grid"], bc_table["I"][wave_idx])
    #                     # if cone_half_angular_width is None:
    #                     #     Ibc = weno4(mu_hit, bc_table["mu_grid"], bc_table["I"][20])
    #                     # else:
    #                     #     Ibc = 4 / 9 * weno4(mu_hit, bc_table["mu_grid"], bc_table["I"][20])
    #                     #     Ibc += 2.5 / 9 * weno4(mu_hit - gl_sample * cone_half_angular_width, bc_table["mu_grid"], bc_table["I"][20])
    #                     #     Ibc += 2.5 / 9 * weno4(mu_hit + gl_sample * cone_half_angular_width, bc_table["mu_grid"], bc_table["I"][20])
    #                     # probe_I[probe_v, probe_u] += j_weights[j] * Ibc * abs(mu_x) * 0.5 * np.pi
    #                     probe_I[probe_v, probe_u] += Ibc * abs(mu_x) * 0.5 * np.pi
    #     return probe_I


    # probe_spacing = 16
    # probe_I = np.zeros((32, 32))
    # probe_hits = np.zeros((32, 32))
    # cone_half_angular_width = 2.0 * np.pi / 2048
    # # cone_half_angular_width = None
    # for i in tqdm(range(1024)):
    #     probe_I += compute_ray_idx(i, cone_half_angular_width=cone_half_angular_width)
    #     probe_hits += (probe_I != 0.0)


    # probe_I /= 1024
    # probe_I = (probe_I << u.Unit("W / (m2 Hz sr)")).to("kW / (m2 nm sr)", equivalencies=u.spectral_density(wav=dex_wave[20] * u.nm)).value

    # out = netCDF4.Dataset("build/output.nc")
    # casc = out["cascade"][...]
    # plt.ion()
    # plt.clf()

