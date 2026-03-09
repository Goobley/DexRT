#!/bin/bash

NETCDF_INCLUDE_PATH=$(nc-config --includedir)
NETCDF_LIBS=$(nc-config --libs)
NETCDF_LIBDIR=$(nc-config --libdir)

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
cmake \
      -DDEX_ARCH="HIP"              \
      -DYAKL_AUTO_PROFILE="On"         \
      -DDEX_CXX_FLAGS="-O3 -std=gnu++20 -Wno-unused-value" \
      -DCMAKE_CXX_COMPILER="hipcc" \
      -DNETCDF_INCLUDE_PATH="${NETCDF_INCLUDE_PATH}" \
      -DLDLIBS="${NETCDF_LIBS}" \
      -DLDFLAGS="-Wl,-rpath,${NETCDF_LIBDIR}" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../kokkos-kernels" \
      -DCMAKE_BUILD_TYPE="Release" \
      ..

