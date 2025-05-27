#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
NETCDF_INCLUDE_PATH=$(../docker_deps/netcdf/install/bin/nc-config --includedir)
NETCDF_LIBS=$(../docker_deps/netcdf/install/bin/nc-config --libs)
NETCDF_LIBDIR=$(../docker_deps/netcdf/install/bin/nc-config --libdir)

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
cmake \
      -DDEX_ARCH="CUDA"              \
      -DYAKL_AUTO_PROFILE="On"         \
      -DDEX_CXX_FLAGS="-O3 --generate-line-info -ftz=true -std=c++20" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="${NETCDF_INCLUDE_PATH}" \
      -DLDLIBS="${NETCDF_LIBS}" \
      -DLDFLAGS="-Wl,-rpath,${NETCDF_LIBDIR}" \
      -DKokkos_ROOT="$(pwd)/../kokkos/" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../kokkos/lib64/cmake;$(pwd)/../kokkos-kernels/lib64/cmake/" \
      -DCMAKE_BUILD_TYPE="Release" \
      ..

