#!/bin/bash

module purge
export NVHPC_CUDA_HOME="/opt/software/nvidia/sdk/Linux_x86_64/24.5/cuda/12.4/"
module load gcc/12.2.0
module load mkl/2022.1.0
module load nvidia/sdk/24.5
module load magma/gcc-8.5.0/2.8.0
module load netcdf/gcc-8.5.0/4.9.0
