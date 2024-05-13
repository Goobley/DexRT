import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import astropy.units as u
import lightweaver as lw
from lightweaver.rh_atoms import CaII_atom, H_6_atom
import promweaver as pw
from stratified_prom import StratifiedPromModel

def collisionless_CaII():
    ca = CaII_atom()
    ca.collisions = []
    lw.reconfigure_atom(ca)
    return ca

def nothing_H():
    h = H_6_atom()
    h.lines = []
    h.continua = []
    lw.reconfigure_atom(h)
    return h

def nothing_CaII():
    ca = CaII_atom()
    ca.lines = []
    ca.continua = []
    lw.reconfigure_atom(ca)
    return ca

if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    dex_J = np.array(ds["image"][...])
    dex_wave = np.array(ds["wavelength"][...])
    dex_eta = np.array(ds["eta"][...])
    dex_chi = np.array(ds["chi"][...])
    dex_pops = np.array(ds["pops"][...])

    dex_atmos = netCDF4.Dataset("build/atmos.nc", "r")
    nz = dex_atmos.dimensions["z"].size
    z_grid = np.ascontiguousarray((np.arange(nz, dtype=np.float64) * dex_atmos["voxel_scale"][...])[::-1])
    sample_idx = dex_atmos["ne"].shape[1] // 2
    temperature = np.ascontiguousarray(dex_atmos["temperature"][:, sample_idx][::-1]).astype(np.float64)
    ne = np.ascontiguousarray(dex_atmos["ne"][:, sample_idx][::-1]).astype(np.float64)
    nhtot = np.ascontiguousarray(dex_atmos["nh_tot"][:, sample_idx][::-1]).astype(np.float64)
    vturb = np.ascontiguousarray(dex_atmos["vturb"][:, sample_idx][::-1]).astype(np.float64)
    pressure = np.ascontiguousarray(dex_atmos["pressure"][:, sample_idx][::-1]).astype(np.float64)

    bc_ctx = pw.compute_falc_bc_ctx(["H", "Ca"])
    # bc_table = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.1, 1.0, 10))
    bc_table = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20))
    # bc_table = pw.tabulate_bc(bc_ctx)
    I_with_zero = np.zeros((bc_table["I"].shape[0], bc_table["I"].shape[1] + 1))
    I_with_zero[:, 1:] = bc_table["I"][...]
    bc_table["I"] = I_with_zero
    bc_table["mu_grid"] = np.concatenate([[0], bc_table["mu_grid"]])
    bc_provider = pw.TabulatedPromBcProvider(**bc_table)


    model = StratifiedPromModel(
        "filament",
        z=z_grid,
        temperature=temperature,
        vlos=np.zeros_like(ne),
        ne=ne,
        pressure=pressure,
        vturb=vturb,
        altitude=np.float64(dex_atmos["offset_z"][...]),
        active_atoms=["H", "Ca"],
        Nrays=10,
        BcType=pw.ConePromBc,
        bc_provider=bc_provider,
        ctx_kwargs={"formalSolver": "piecewise_linear_1d"},
        Nthreads=12,
        do_pressure_updates=True,
    )
    model.ctx.background.eta[...] = 0.0
    model.ctx.background.chi[...] = 0.0
    model.ctx.background.sca[...] = 0.0
    model.iterate_se(popsTol=1e-2, JTol=1.0, printInterval=0.0)
    ctx = model.ctx

    slice_idx = 128
    J_slice = (ctx.spect.J[:, ctx.spect.J.shape[1] - slice_idx - 1] << u.Unit("W / (m2 Hz sr)")).to("kW / (m2 nm sr)", equivalencies=u.spectral_density(ctx.spect.wavelength * u.nm))

    plt.ion()
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(dex_wave, dex_J[:, slice_idx, sample_idx], '-+', label="J Dex")
    ax[0].plot(ctx.spect.wavelength, J_slice, '-+', label="J Lightweaver")
    ax[0].set_xlabel(r"$\lambda$ [nm]")
    ax[0].set_ylabel(r"$J$ [kW / (m2 nm sr)]")
    ax[0].legend()
    dex_interp = np.interp(ctx.spect.wavelength, dex_wave, dex_J[:, slice_idx, sample_idx])
    ax[1].plot(ctx.spect.wavelength, dex_interp / J_slice, label="dex / lw")
    ax[1].set_xlabel(r"$\lambda$ [nm]")
    # ax[1].set_ylim(0.70)
    ax[1].legend()
