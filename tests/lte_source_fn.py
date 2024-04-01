import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, CaII_atom
import numpy as np
import astropy.units as u

if __name__ == "__main__":
    atmos = Falc82()
    atmos.quadrature(3)

    extra_wave = CaII_atom().lines[-1].lambda0 - 1.0

    a_set = lw.RadiativeSet([H_6_atom(), CaII_atom()])
    a_set.set_detailed_static("Ca")
    eq_pops = a_set.compute_eq_pops(atmos)
    spect = a_set.compute_wavelength_grid(extraWavelengths=np.array([extra_wave]))

    ctx = lw.Context(atmos, spect, eq_pops)
    ctx.background.eta[...] = 0.0
    ctx.background.chi[...] = 0.0
    ctx.background.sca[...] = 0.0
    ctx.depthData.fill = True
    ctx.formal_sol_gamma_matrices()

    sample_idx = 30
    wavelength_idx = 100
    eta =  []
    chi = []
    wave = []
    wavelength_idx = np.searchsorted(ctx.spect.wavelength, CaII_atom().lines[0].lambda0)
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx])
    wave.append(ctx.spect.wavelength[wavelength_idx])
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx+1])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx+1])
    wave.append(ctx.spect.wavelength[wavelength_idx])
    # lambda_0[8542] - 1.0 nm
    wavelength_idx = np.searchsorted(ctx.spect.wavelength, extra_wave)
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx])
    wave.append(ctx.spect.wavelength[wavelength_idx])
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx-1])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx-1])
    wave.append(ctx.spect.wavelength[wavelength_idx])
    # lambda_0[8542]
    wavelength_idx = np.searchsorted(ctx.spect.wavelength, CaII_atom().lines[-1].lambda0)
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx-1])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx-1])
    wave.append(ctx.spect.wavelength[wavelength_idx])
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx])
    wave.append(ctx.spect.wavelength[wavelength_idx])
    # Continuum @ 75 nm
    wavelength_idx = np.searchsorted(ctx.spect.wavelength, 75.0)
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx])
    wave.append(ctx.spect.wavelength[wavelength_idx])
    eta.append(ctx.depthData.eta[wavelength_idx, 0, 0, sample_idx+1])
    chi.append(ctx.depthData.chi[wavelength_idx, 0, 0, sample_idx+1])
    wave.append(ctx.spect.wavelength[wavelength_idx])

    wave = np.asarray(wave) << u.nm
    eta = np.asarray(eta) << u.Unit("W / (m3 Hz sr)")
    chi = np.asarray(chi) << u.Unit("m-1")

    eta_dex = (eta * (1 * u.m)).to("kW / (m2 nm sr)", equivalencies=u.spectral_density(wav=wave)) / (1 * u.m)
    chi_dex = chi

    def to_cpp_repr(arr):
        arr_repr = "{"
        for i, e in enumerate(arr.value):
            arr_repr += f"FP({e:g})"
            if i != arr.shape[0] - 1:
                arr_repr += ", "
        arr_repr += "}"
        return arr_repr

    eta_repr = to_cpp_repr(eta_dex)
    chi_repr = to_cpp_repr(chi_dex)

    print(f"constexpr fp_t eta_expec[] = {eta_repr};")
    print(f"constexpr fp_t chi_expec[] = {chi_repr};")




