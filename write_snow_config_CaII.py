from dexrt_py.dexrt.config_schemas.dexrt import DexrtNonLteConfig, AtomicModelConfig, DexrtLteConfig
from dexrt_py.dexrt.write_config import write_config

if __name__ == "__main__":
    config_sparse = DexrtNonLteConfig(
        atmos_path="snow_atmos_steeper_10Mm.nc",
        output_path="snow_khi_10Mm_steeper_CaII.nc",
        sparse_calculation=True,
        atoms={
            "Ca": AtomicModelConfig(
                path="../tests/test_CaII.yaml"
            )
        },
        boundary_type="Promweaver",
        conserve_charge=False,
        conserve_pressure=False,
    )

    write_config(config_sparse, "build/snow_khi_sparse.yaml")
    # 18 iter: 12m10s

    config_dense = DexrtNonLteConfig(
        atmos_path="snow_atmos_steeper_10Mm.nc",
        output_path="snow_khi_10Mm_steeper_CaII_dense.nc",
        sparse_calculation=False,
        atoms={
            "Ca": AtomicModelConfig(
                path="../tests/test_CaII.yaml"
            )
        },
        boundary_type="Promweaver",
        conserve_charge=False,
        conserve_pressure=False,
    )

    write_config(config_dense, "build/snow_khi_dense.yaml")
    # 18 iter: 20m43s

    config_lte = DexrtLteConfig(
        atmos_path="snow_atmos_steeper_10Mm.nc",
        output_path="snow_khi_10Mm_steeper_CaII_lte.nc",
        atoms={
            "Ca": AtomicModelConfig(
                path="../tests/test_CaII.yaml"
            )
        },
        boundary_type="Promweaver",
    )
    write_config(config_lte, "build/snow_khi_lte.yaml")