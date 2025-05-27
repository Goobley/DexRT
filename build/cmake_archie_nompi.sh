#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
NVJIT_PATH="/opt/software/nvidia/sdk/Linux_x86_64/24.5/cuda/12.4/targets/x86_64-linux/lib/"

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v -ftz=true" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
      # -DYAKL_VERBOSE="On" \
cmake  \
      -DDEX_ARCH="CUDA"              \
      -DYAKL_AUTO_PROFILE="On"         \
      -DDEX_CXX_FLAGS="-O3 --generate-line-info -ftz=true" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs) -lnvJitLink" \
      -DKokkos_ROOT="$(pwd)/../kokkos/" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../kokkos/lib64/cmake;$(pwd)/../kokkos-kernels/lib64/cmake/" \
      -DCMAKE_BUILD_TYPE="Release" \
      ..
