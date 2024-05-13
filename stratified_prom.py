from typing import Dict, List, Optional, Type

import lightweaver as lw
import lightweaver.wittmann as witt
import numpy as np

from promweaver.bc_provider import DynamicContextPromBcProvider
from promweaver.compute_bc import compute_falc_bc_ctx
from promweaver.j_prom_bc import UniformJPromBc
from promweaver.prom_bc import PromBc
from promweaver.prom_model import PromModel
from promweaver.utils import default_atomic_models


class StratifiedPromModel(PromModel):
    """
    Class for "Iso" prominence simulations. Iso implies isothermal and isobaric.

    Parameters
    ----------
    projection : str
        Whether the object is to be treated as a "filament" or a "prominence".
    z : array
        Height stratification (usual RT upside-down)
    temperature : array
        The temperature stratification of the prominence [K].
    ne : array
        Electron density stratification [m-3]
    vlos : array
        The z-projected velocity stratification to apply to the prominence model [m/s].
    pressure : array
        The pressure stratification of the prominence [Pa].
    vturb : array
        The microturbulent velocity inside the prominence [m/s].
    altitude : float
        The altitude of the prominence above the solar surface [m].
    active_atoms : list of str
        The element names to make "active" i.e. consider in non-LTE.
    detailed_atoms : list of str, optional
        The element names to make "detailed static" i.e. consider in detail
        radiatively.  If 'H' is set detailed then the model will be run with a
        fixed electron density (set to balance pressure) and iterative charge
        conservation disabled. Provide a compatible `bc_provider` as these will
        not be taken account in the default one computed on the fly.
    detailed_pops : dict, optional
        A mapping from names in `detailed_atoms` to an array of their population
        distribution of shape suitable for the model.
    atomic_models : list of `lw.AtomicModels`, optional
        The atomic models to use, a default set will be chosen if none are
        specified.
    Nrays : int, optional
        The number of Gauss-Legendre angular quadrature rays to use in the
        model. Default: 3. This number will need to be set higher (e.g. 10) if
        using the `ConePromBc`.
    Nthreads : int, optional
        The number of CPU threads to use when solving the radiative transfer
        equations. Default: 1.
    prd : bool, optional
        Whether to consider the effects of partial frequency redistribution.
        Default: False.
    vrad : float, optional
        The Doppler-dimming radial velocity to apply. Note that for filaments
        this is the same as `vlos` and that should be used instead. Not fully
        supported in boundary conditions yet, (i.e. you should interpolate the
        wavelength grid first). Default: None, i.e. 0.
    ctx_kwargs : dict, optional
        Extra kwargs to be passed when constructing the Context.
    BcType : Constructor for a type of PromBc, optional
        The base type to be used for constructing the boundary conditions.
        Default: UniformJPromBc.
    bc_kwargs : dict, optional
        Extra kwargs to be passed to the construction of the boundary conditions.
    bc_provider : PromBcProvider, optional
        The provider to use for computing the radiation in the boundary
        conditions. Default: `DynamicContextPromBcProvider` using an
        `lw.Context` configured to match the current model. Note that the
        default is note very performant, but is convenient for experimenting.
        When running a grid of models, consider creating a
        `TabulatedPromBcProvider` using `compute_falc_bc_ctx` and `tabulate_bc`,
        since the default performs quite a few extra RT calculations.
    """

    def __init__(
        self,
        projection,
        z,
        temperature,
        ne,
        vlos,
        pressure,
        vturb,
        altitude,
        active_atoms: List[str],
        detailed_atoms: List[str] = None,
        detailed_pops: Optional[Dict[str, np.ndarray]] = None,
        nH = False,
        atomic_models=None,
        Nrays=3,
        Nthreads=1,
        prd=False,
        vrad: Optional[float] = None,
        ctx_kwargs=None,
        BcType: Optional[Type[PromBc]] = None,
        bc_kwargs=None,
        bc_provider=None,
        do_pressure_updates = False,
    ):
        self.projection = projection
        if projection not in ["prominence", "filament"]:
            raise ValueError(
                f"Expected projection ({projection}), to be 'prominence' or 'filament'"
            )

        self.z = z
        self.temperature = temperature
        self.ne = ne
        self.nH = nH
        self.vlos = vlos
        self.pressure = pressure
        self.vturb = vturb
        self.altitude = altitude
        self.Nrays = Nrays
        self.prd = prd
        self.vrad = vrad
        self.do_pressure_updates = do_pressure_updates
        # NOTE(cmo): Whether to do pressure/n_e updates, only disabled for detailed H
        self.do_pressure_updates = True

        if projection == "filament" and vrad is not None and vlos is not None:
            raise ValueError(
                "Cannot set both vrad and vlos for a filament. (Just set one of the two)."
            )

        if ctx_kwargs is None:
            ctx_kwargs = {}
        if BcType is None:
            BcType = UniformJPromBc
        if bc_kwargs is None:
            bc_kwargs = {}

        if atomic_models is None:
            atomic_models = default_atomic_models()

        if bc_provider is None:
            vz = None
            if projection == "prominence" and vrad is not None:
                vz = self.vrad

            ctx = compute_falc_bc_ctx(
                active_atoms=active_atoms,
                atomic_models=atomic_models,
                prd=self.prd,
                vz=vz,
                Nthreads=Nthreads,
            )
            bc_provider = DynamicContextPromBcProvider(ctx)

        lower_bc = BcType(projection, bc_provider, altitude, "lower", **bc_kwargs)
        if projection == "prominence":
            upper_bc: Optional[PromBc] = BcType(
                projection, bc_provider, altitude, "upper", **bc_kwargs
            )
        else:
            upper_bc = None

        if isinstance(nH, np.ndarray):
            print("Using nHTot from model..")
            nHTot = nH
        else:
            print("Calculating nHTot...")
            nHTot = pressure / (lw.KBoltzmann * temperature) - ne

        self.atmos = lw.Atmosphere.make_1d(
            lw.ScaleType.Geometric,
            depthScale=self.z,
            temperature=self.temperature,
            vlos=self.vlos,
            vturb=self.vturb,
            ne=self.ne,
            nHTot=nHTot,
            lowerBc=lower_bc,
            upperBc=upper_bc,
        )
        self.atmos.quadrature(Nrays)


        self.rad_set = lw.RadiativeSet(atomic_models)
        self.rad_set.set_active(*active_atoms)
        if detailed_atoms is not None:
            if detailed_pops is None:
                detailed_pops = {}
            self.rad_set.set_detailed_static(*detailed_atoms)
            elements = [lw.PeriodicTable[a] for a in detailed_atoms]
            if lw.PeriodicTable['H'] in elements:
                # NOTE(cmo): Can't do charge conservation
                self.do_pressure_updates = False

            # NOTE(cmo): Set ne now to correctly match pressure if "H" pops are provided
            if "H" in detailed_pops:
                new_nHTot = np.sum(detailed_pops["H"], axis=0)
            else:
                self.eq_pops = self.rad_set.iterate_lte_ne_eq_pops(self.atmos)
                new_nHTot = np.sum(self.eq_pops["H"], axis=0)

            new_ne = (
                self.pressure / (lw.KBoltzmann * self.temperature)
                - lw.DefaultAtomicAbundance.totalAbundance * new_nHTot
            )
            self.atmos.nHTot[:] = new_nHTot
            self.atmos.ne[:] = new_ne

            if "H" in detailed_pops:
                self.eq_pops = self.rad_set.compute_eq_pops(self.atmos)

            for ele, pops in detailed_pops.items():
                if ele not in detailed_atoms:
                    continue
                self.eq_pops[ele][...] = pops
        else:
            self.eq_pops = self.rad_set.iterate_lte_ne_eq_pops(self.atmos)

        self.spect = self.rad_set.compute_wavelength_grid()
        self.Nthreads = Nthreads
        hprd = self.prd and self.vlos is not None
        if hprd and hprd not in ctx_kwargs:
            ctx_kwargs["hprd"] = hprd
        ctx = lw.Context(
            self.atmos,
            self.spect,
            self.eq_pops,
            Nthreads=Nthreads,
            conserveCharge=self.do_pressure_updates,
            **ctx_kwargs,
        )
        super().__init__(ctx)

    def iterate_se(self, *args, update_model_kwargs: Optional[dict] = None, **kwargs):
        if update_model_kwargs is None:
            update_model_kwargs = {}

        if self.prd and "prd" not in kwargs:
            kwargs["prd"] = self.prd

        update_model = None
        #self.do_pressure_updates = False
        if self.do_pressure_updates:
            def update_model(self, printNow, **kwargs):
                # NOTE(cmo): Fix pressure throughout the atmosphere.
                N = (
                    lw.DefaultAtomicAbundance.totalAbundance * self.atmos.nHTot
                    + self.atmos.ne
                )
                NError = self.pressure / (lw.KBoltzmann * self.temperature) - N
                nHTotCorrection = NError / (
                    lw.DefaultAtomicAbundance.totalAbundance
                    + self.eq_pops["H"][-1] / self.atmos.nHTot
                )
                if printNow:
                    print(
                        f"    nHTotError: {np.max(np.abs(nHTotCorrection / self.atmos.nHTot))}"
                    )
                self.atmos.ne[:] += (
                    nHTotCorrection * self.eq_pops["H"][-1] / self.atmos.nHTot
                )
                prevnHTot = np.copy(self.atmos.nHTot)
                self.atmos.nHTot[:] += nHTotCorrection
                if np.any(self.atmos.nHTot < 0.0):
                    raise lw.ConvergenceError("nHTot driven negative!")
                nHTotRatio = self.atmos.nHTot / prevnHTot

                for atom in self.rad_set.activeAtoms:
                    p = self.eq_pops[atom.element]
                    p[...] *= nHTotRatio[None, :]

                # TODO(cmo): Add condition to not always re-evaluate the line profiles. Maybe on ne change?
                self.ctx.update_deps(vlos=False, background=False)

        return super().iterate_se(
            *args,
            update_model=update_model,
            update_model_kwargs=update_model_kwargs,
            **kwargs,
        )
