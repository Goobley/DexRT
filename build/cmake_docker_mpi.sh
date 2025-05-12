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
cmake \
      -DDEX_ARCH="CUDA"              \
      -DYAKL_AUTO_PROFILE="On"         \
      -DDEX_CXX_FLAGS="-O3 --generate-line-info -ftz=true" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DDEXRT_USE_MPI="On" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs)" \
      -DKokkos_ROOT="$(pwd)/../kokkos/" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../kokkos/lib/cmake;$(pwd)/../kokkos-kernels/lib/cmake/" \
      -DCMAKE_BUILD_TYPE="Release" \
      ..

# run as
# mpirun -mca btl_smcuda_use_cuda_ipc 0 -np 2 ./dexrt
# when running on one gpu. Cuda ipc is frequently causing issues where it gets stuck and the array gets filled with zeros during a broadcast?? Probably not an issue with multiple cards?
# problem seems similar to: https://github.com/open-mpi/ompi/issues/6001 (albeit on same GPU)
# See also: https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html#can-i-get-additional-cuda-debug-level-information-at-run-time
# "In addition, it is assumed that CUDA IPC is possible when running on the same GPU, and this is typically true." In this instance, it was not.