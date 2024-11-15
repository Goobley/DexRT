#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MAGMA_INCLUDE_PATH="/usr/local/magma/include"
MAGMA_LIB_PATH="${MAGMA_INCLUDE_PATH}/../lib/"
MAGMA_LIB="-lmagma"

# NOTE(cmo): You will need to get this into your runenv too
module purge
module load nvhpc-openmpi3/24.5

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v -ftz=true" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
      # -DYAKL_VERBOSE="On" \
      # -DMPI_INCLUDE_PATH="${MPI_INCLUDE_PATH}" \
cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_CUDA_COMPILER=nvcc    \
      -DCMAKE_C_COMPILER=gcc         \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="CUDA"              \
      -DYAKL_AUTO_PROFILE="On"         \
      -DYAKL_CUDA_FLAGS="-O0 -g --ptxas-options=-v --generate-line-info -ftz=true" \
      -DYAKL_INT64_RESHAPE="On"       \
      -DDEXRT_CUDA_ARCHITECTURES="86" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DDEXRT_USE_MPI="On" \
      -DMAGMA_INCLUDE_PATH="${MAGMA_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs) -L${MAGMA_LIB_PATH} ${MAGMA_LIB}" \
      ..

# run as
# mpirun -mca btl_smcuda_use_cuda_ipc 0 -np 2 ./dexrt
# when running on one gpu. Cuda ipc is frequently causing issues where it gets stuck and the array gets filled with zeros during a broadcast?? Probably not an issue with multiple cards?
