from typing import Dict, Literal, Union, List
from pydantic import BaseModel, Field

class AtomicModelConfig(BaseModel):
    path: str
    treatment: Union[Literal["Detailed"], Literal["Golding"], Literal["Active"]] = "Active"

class DexrtSystemConfig(BaseModel):
    mem_pool_initial_gb: float = 2.0
    mem_pool_grow_gb: float = 1.4

class DexrtOutputConfig(BaseModel):
    sparse: bool = False
    wavelength: bool = True
    J: bool = True
    pops: bool = True
    lte_pops: bool = True
    ne: bool = True
    nh_tot: bool = True
    max_mip_level = True
    alo: bool = False
    cascades: List[int] = Field(default_factory=list)

class DexrtConfig(BaseModel):
    system: DexrtSystemConfig = Field(default_factory=DexrtSystemConfig)
    atmos_path: str = "dexrt_atmos.nc"
    output_path: str = "dexrt.nc"
    mode: Union[Literal["Lte"], Literal["NonLte"], Literal["GivenFs"]] = "NonLte"
    store_J_on_cpu: bool = True
    output: DexrtOutputConfig = Field(default_factory=DexrtOutputConfig)

class DexrtLteConfig(DexrtConfig):
    mode: Literal["Lte"] = "Lte"
    sparse_calculation: bool = False
    threshold_temperature: float = 250e3
    atoms: Dict[str, AtomicModelConfig]
    boundary_type: Union[Literal["Zero"], Literal["Promweaver"]]

class DexrtNonLteConfig(DexrtLteConfig):
    mode: Literal["NonLte"] = "NonLte"
    max_iter: int = 200
    pop_tol: float = 1e-3
    conserve_charge: bool = False
    conserve_pressure: bool = False
    snapshot_frequency: int = 0
    initial_lambda_iterations: int = 2

class DexrtGivenFsConfig(DexrtConfig):
    mode: Literal["GivenFs"] = "GivenFs"
