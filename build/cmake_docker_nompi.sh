#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MPI_INCLUDE_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/include"

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v" \
cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_CUDA_COMPILER=nvcc    \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="CUDA"              \
      -DYAKL_PROFILE="On"             \
      -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v --generate-line-info" \
      -DDEXRT_CUDA_ARCHITECTURES="86" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DMPI_INCLUDE_PATH="${MPI_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDFLAGS="$(nc-config --libs)" \
      ..
