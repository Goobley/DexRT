from dexrt_py.dexrt.config_schemas.dexrt import DexrtNonLteConfig, DexrtLteConfig, DexrtGivenFsConfig
import yaml

def write_config(conf, path="dexrt.yaml"):
    data = conf.model_dump()
    with open(path, "w") as f:
        yaml.dump(data, f)