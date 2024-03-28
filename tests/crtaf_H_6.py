import crtaf
from crtaf.from_lightweaver import LightweaverAtomConverter
from lightweaver.rh_atoms import H_6_atom
import numpy as np

def make_atom():
    conv = LightweaverAtomConverter()
    model = conv.convert(H_6_atom())
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)
    return model_simplified
if __name__ == "__main__":
    atom = make_atom()
    with open("H_6.yaml", "w") as f:
        f.write(atom.yaml_dumps())
