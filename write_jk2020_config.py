from dexrt_py.dexrt.config_schemas.dexrt import DexrtNonLteConfig, AtomicModelConfig, DexrtLteConfig, DexrtSystemConfig
from dexrt_py.dexrt.write_config import write_config

if __name__ == "__main__":
    step_idx = 550
    config_sparse = DexrtNonLteConfig(
        atmos_path=f"jk2020{step_idx:04d}_dex.nc",
        output_path=f"jk2020{step_idx:04d}_synth.nc",
        sparse_calculation=True,
        threshold_temperature=250e3,
        atoms={
            "H": AtomicModelConfig(
                path="../tests/H_6.yaml"
            ),
            "Ca": AtomicModelConfig(
                path="../tests/test_CaII.yaml"
            )
        },
        boundary_type="Promweaver",
        conserve_charge=True,
        conserve_pressure=True,
        max_iter=1000,
        store_J_on_cpu=True,
        system=DexrtSystemConfig(
            mem_pool_initial_gb=12,
            mem_pool_grow_gb=5.9
        )
    )

    write_config(config_sparse, f"jk2020{step_idx:04d}.yaml")
