import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import astropy.units as u
import lightweaver as lw
from lightweaver.rh_atoms import CaII_atom, H_6_atom

def collisionless_CaII():
    ca = CaII_atom()
    ca.collisions = []
    lw.reconfigure_atom(ca)
    return ca

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

    atmos = lw.Atmosphere.make_1d(
        lw.ScaleType.Geometric,
        z_grid,
        temperature,
        vlos=np.zeros_like(z_grid),
        vturb=vturb,
        ne=ne,
        nHTot=nhtot,
        lowerBc=lw.ZeroRadiation(),
        upperBc=lw.ZeroRadiation(),
    )
    atmos.quadrature(5)

    # a_set = lw.RadiativeSet([H_6_atom(), collisionless_CaII()])
    a_set = lw.RadiativeSet([H_6_atom(), CaII_atom()])
    a_set.set_active("Ca")
    eq_pops = a_set.compute_eq_pops(atmos)
    spect = a_set.compute_wavelength_grid(lambdaReference=CaII_atom().lines[-1].lambda0 - 1.0)

    ctx = lw.Context(atmos, spect, eq_pops, formalSolver="piecewise_linear_1d")
    ctx.background.eta[...] = 0.0
    ctx.background.chi[...] = 0.0
    ctx.background.sca[...] = 0.0
    ctx.depthData.fill = True
    ctx.formal_sol_gamma_matrices()
    # lw.iterate_ctx_se(ctx, popsTol=5e-2, JTol=1.0)
    slice_idx = 128
    J_slice = (ctx.spect.J[:, slice_idx] << u.Unit("W / (m2 Hz sr)")).to("kW / (m2 nm sr)", equivalencies=u.spectral_density(ctx.spect.wavelength * u.nm))

    plt.ion()
    fig, ax = plt.subplots(2, 1)
    ax[0].semilogy(dex_wave, dex_J[:, slice_idx, sample_idx], '-+', label="J Dex")
    ax[0].semilogy(ctx.spect.wavelength, J_slice, '-+', label="J Lightweaver")
    ax[0].set_xlabel(r"$\lambda$ [nm]")
    ax[0].set_ylabel(r"$J$ [kW / (m2 nm sr)]")
    ax[0].legend()
    dex_interp = np.interp(ctx.spect.wavelength, dex_wave, dex_J[:, slice_idx, sample_idx])
    ax[1].plot(ctx.spect.wavelength, dex_interp / J_slice, label="dex / lw")
    ax[1].set_xlabel(r"$\lambda$ [nm]")
    ax[1].set_ylim(0.70)
    ax[1].legend()
