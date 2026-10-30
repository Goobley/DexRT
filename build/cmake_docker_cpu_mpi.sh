#!/bin/bash
# CPU (OpenMP) + MPI build, e.g. for testing the domain decomposed SC solver.
# Uses the OpenMP-only Kokkos installs (kokkos_omp, kokkos-kernels_omp). Run
# from a separate build directory, e.g. build_cpu_mpi/.

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MPI_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/mpi
DEX_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

rm -rf CMakeCache.txt CMakeFiles

cmake \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      -DDEX_CXX_FLAGS="-O3 -fopenmp" \
      -DDEXRT_USE_MPI="On" \
      -DMPI_HOME="${MPI_ROOT}" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs) -fopenmp" \
      -DKokkos_ROOT="${DEX_ROOT}/kokkos_omp/" \
      -DCMAKE_PREFIX_PATH="${DEX_ROOT}/kokkos_omp/lib/cmake;${DEX_ROOT}/kokkos-kernels_omp/lib/cmake/" \
      -DCMAKE_BUILD_TYPE="Release" \
      -DYAKL_AUTO_PROFILE="On" \
      -DFETCHCONTENT_SOURCE_DIR_ARGPARSE="${DEX_ROOT}/build/_deps/argparse-src" \
      "${DEX_ROOT}"
