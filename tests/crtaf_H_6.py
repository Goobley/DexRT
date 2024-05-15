import crtaf
from crtaf.from_lightweaver import LightweaverAtomConverter
from crtaf.core_types import TemperatureInterpolationRateImpl
from lightweaver.rh_atoms import H_6_atom, CaII_atom, H_4_atom
import numpy as np
import astropy.units as u

def make_atom():
    conv = LightweaverAtomConverter()
    model = conv.convert(H_6_atom())
    for l in model.lines:
        l.wavelength_grid.q_core *= 3
        l.wavelength_grid.q_wing *= 2
        l.wavelength_grid.n_lambda //= 2
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)
    new_lines = []
    for l in model_simplified.lines:
        if l.lambda0 < 1000.0 * u.nm:
            new_lines.append(l)
    model_simplified.lines = new_lines
    return model_simplified

def make_H_4():
    conv = LightweaverAtomConverter()
    model = conv.convert(H_4_atom())
    for l in model.lines:
        l.wavelength_grid.q_core *= 3
        l.wavelength_grid.q_wing *= 2
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)
    return model_simplified

def make_CaII():
    conv = LightweaverAtomConverter()
    model = conv.convert(CaII_atom())
    for l in model.lines:
        l.wavelength_grid.q_core *= 3
        l.wavelength_grid.q_wing *= 2
        l.wavelength_grid.n_lambda //= 2
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)

    current_grid = model_simplified.lines[-1].wavelength_grid.wavelengths
    new_grid = np.sort(np.concatenate((current_grid, [-1.0 * u.nm])))
    model_simplified.lines[-1].wavelength_grid.wavelengths = new_grid

    # NOTE(cmo): To prevent explosion due to rates in the Snow KHI model
    # TODO(cmo): Grab the rates from source/RADYN
    for trans in model_simplified.collisions:
        for coll in trans.data:
            if isinstance(coll, TemperatureInterpolationRateImpl) and coll.temperature[0] > (1000.0 * u.K):
                coll.temperature = np.concatenate(([500.0 * u.K], coll.temperature))
                coll.data = np.concatenate(([0.0 * coll.data.unit], coll.data))
    return model_simplified


if __name__ == "__main__":
    atom = make_atom()
    with open("H_6.yaml", "w") as f:
        f.write(atom.yaml_dumps())

    atom = make_CaII()
    with open("test_CaII.yaml", "w") as f:
        f.write(atom.yaml_dumps())

    atom = make_H_4()
    with open("H_4.yaml", "w") as f:
        f.write(atom.yaml_dumps())