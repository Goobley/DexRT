#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MPI_INCLUDE_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/include"

./cmake_clean.sh

cmake -DDEX_ARCH="CUDA"              \
      -DDEX_CXX_FLAGS="-O0 -g -G" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DMPI_INCLUDE_PATH="${MPI_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDFLAGS="$(nc-config --libs)" \
      -DKokkos_ROOT="$(pwd)/../kokkos-debug/" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../../kokkos-debug/lib/cmake/;$(pwd)/../../kokkos-kernels/lib/cmake" \
      -DCMAKE_BUILD_TYPE="Debug" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DDEX_FORCE_DOUBLE_PREC=Off \
      -DDEX_HALF_RADIATION_STORAGE=On \
      ..