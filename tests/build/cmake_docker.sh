#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MPI_INCLUDE_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/include"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_CUDA_COMPILER=nvcc   \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="CUDA"              \
      -DYAKL_CUDA_FLAGS="-O0 --ptxas-options=-v" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DMPI_INCLUDE_PATH="${MPI_INCLUDE_PATH}" \
      -DLDFLAGS="" \
      ..