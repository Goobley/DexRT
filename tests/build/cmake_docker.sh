#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MPI_INCLUDE_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/include"
MAGMA_INCLUDE_PATH="/usr/local/magma/include"
MAGMA_LIB_PATH="${MAGMA_INCLUDE_PATH}/../lib/"
MAGMA_LIB="-lmagma"

./cmake_clean.sh

cmake -DDEX_ARCH="CUDA"              \
      -DDEX_CXX_FLAGS="-O0 -g -G" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DMPI_INCLUDE_PATH="${MPI_INCLUDE_PATH}" \
      -DMAGMA_INCLUDE_PATH="${MAGMA_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDFLAGS="$(nc-config --libs) -L${MAGMA_LIB_PATH} ${MAGMA_LIB}" \
      -DKokkos_ROOT="$(pwd)/../kokkos-debug/" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../../kokkos-debug/lib/cmake/;$(pwd)/../../kokkos-kernels/lib/cmake" \
      -DCMAKE_BUILD_TYPE="Debug" \
      ..