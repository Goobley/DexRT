from typing import Dict, Literal, Union, List
from pydantic import BaseModel, Field

class DexrtRaySystemConfig(BaseModel):
    mem_pool_initial_gb: float = 2.0
    mem_pool_grow_gb: float = 1.4

class DexrtRayConfig(BaseModel):
    system: DexrtRaySystemConfig = Field(default_factory=DexrtRaySystemConfig)
    dexrt_config_path: str = "dexrt.yaml"
    ray_output_path: str = "ray_output.nc"
    muz: List[float] = Field(default_factory=list)
    mux: List[float] = Field(default_factory=list)
    wavelength: List[float] = Field(default_factory=list)
    rotate_aabb: bool = True
    output_cfn: bool = False
    output_eta_chi: bool = False
