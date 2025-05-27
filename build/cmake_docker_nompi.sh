#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v -ftz=true" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
      # -DYAKL_VERBOSE="On" \
      # We disable warning 174 (Expression has no effect) to avoid warnings
      # related to JasUse, which is in turn necessary to avoid the error related
      # to "first variable use in constexpr-if environment".
cmake  \
      -DDEX_ARCH="CUDA"              \
      -DYAKL_AUTO_PROFILE="On"         \
      -DDEX_CXX_FLAGS="-O3 --generate-line-info -ftz=true -diag-suppress=174" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs)" \
      -DKokkos_ROOT="$(pwd)/../kokkos/" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../kokkos/lib/cmake;$(pwd)/../kokkos-kernels/lib/cmake/" \
      -DCMAKE_BUILD_TYPE="Release" \
      ..
