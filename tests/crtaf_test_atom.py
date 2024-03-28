import crtaf
from crtaf.from_lightweaver import LightweaverAtomConverter
from lightweaver.rh_atoms import H_4_atom
import numpy as np

def make_atom():
    conv = LightweaverAtomConverter()
    h = H_4_atom()
    base_model = conv.convert(H_4_atom())
    base_model_dict = base_model.model_dump()
    base_model_dict["lines"][0]["type"] = "PRD-Voigt"
    base_model_dict["lines"][0]["wavelength_grid"] = {
        "type": "Linear",
        "n_lambda": 101,
        "delta_lambda": {
            "unit": "nm",
            "value": 0.01,
        }
    }

    w = h.continua[-1].wavelength()
    new_cont = {
        "type": "Tabulated",
        "transition": ["ii_1", "i_3"],
        "unit": ["nm", "m^2"],
        "value": np.hstack((w[:, None], h.continua[-1].alpha(w)[:, None])).tolist()
    }
    base_model_dict["continua"][2] = new_cont

    base_model_dict["collisions"][3]["data"].append({
        "type": "ChargeExcH",
        "temperature": base_model_dict["collisions"][3]["data"][0]["temperature"],
        "data": {
            "unit": "m3 s-1",
            "value": base_model_dict["collisions"][3]["data"][0]["data"]["value"],
        }
    })
    base_model_dict["collisions"][4]["data"].append({
        "type": "ChargeExcP",
        "temperature": base_model_dict["collisions"][4]["data"][0]["temperature"],
        "data": {
            "unit": "m3 s-1",
            "value": base_model_dict["collisions"][4]["data"][0]["data"]["value"],
        }
    })
    base_model_dict["collisions"][0]["data"].append({
        "type": "Omega",
        "temperature": base_model_dict["collisions"][0]["data"][0]["temperature"],
        "data": {
            "unit": "",
            "value": [float(x) for x in range(len(base_model_dict["collisions"][0]["data"][0]["data"]["value"]))],
        }
    })
    base_model_dict["collisions"][1]["data"].append({
        "type": "CP",
        "temperature": base_model_dict["collisions"][1]["data"][0]["temperature"],
        "data": {
            "unit": "m3 s-1",
            "value": base_model_dict["collisions"][1]["data"][0]["data"]["value"],
        }
    })
    base_model_dict["collisions"][2]["data"].append({
        "type": "CH",
        "temperature": base_model_dict["collisions"][2]["data"][0]["temperature"],
        "data": {
            "unit": "m3 s-1",
            "value": base_model_dict["collisions"][2]["data"][0]["data"]["value"],
        }
    })

    test_model = crtaf.Atom.model_validate(base_model_dict)
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    test_model_simplified = test_model.simplify_visit(visitor)
    return test_model_simplified


if __name__ == "__main__":
    atom = make_atom()
    with open("test_atom.yaml", "w") as f:
        f.write(atom.yaml_dumps())


