import crtaf
from crtaf.from_lightweaver import LightweaverAtomConverter
from lightweaver.rh_atoms import H_6_atom, CaII_atom
import numpy as np
import astropy.units as u

def make_atom():
    conv = LightweaverAtomConverter()
    model = conv.convert(H_6_atom())
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)
    return model_simplified

def make_CaII():
    conv = LightweaverAtomConverter()
    model = conv.convert(CaII_atom())
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)

    current_grid = model_simplified.lines[-1].wavelength_grid.wavelengths
    new_grid = np.sort(np.concatenate((current_grid, [-1.0 * u.nm])))
    model_simplified.lines[-1].wavelength_grid.wavelengths = new_grid
    return model_simplified


if __name__ == "__main__":
    atom = make_atom()
    with open("H_6.yaml", "w") as f:
        f.write(atom.yaml_dumps())

    atom = make_CaII()
    with open("test_CaII.yaml", "w") as f:
        f.write(atom.yaml_dumps())
