#!/bin/bash

module purge
module load /home/y07/shared/tursa-modules/setup-env

module load gcc/12.2.0
# module load nvhpc/23.5-nompi
# module load openmpi/4.1.5-nvhpc235-cuda12
module load cuda/12.3
module load openmpi/4.1.5-gcc12-cuda12
module load cmake

module list 

NETCDF_PATH=/home/dp407/dp407/dc-osbo3/DexRT/docker_deps/netcdf/install/bin
export PATH="${PATH}:${NETCDF_PATH}"

