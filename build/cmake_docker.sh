#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MPI_INCLUDE_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/include"
# ;/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/include;/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/include;/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/nccl/include;/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/nvshmem/include;/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/extras/qd/include/qd;/usr/local/gdrcopy/include;/usr/include/c++/11;/usr/include/x86_64-linux-gnu/c++/11;/usr/include/c++/11/backward;/usr/lib/gcc/x86_64-linux-gnu/11/include;/usr/local/include;/usr/include/x86_64-linux-gnu;/usr/include"

./cmake_clean.sh

      # -DYAKL_ARCH=""              \
      # -DYAKL_CUDA_FLAGS="-O0 -g --ptxas-options=-v" \
      # -DYAKL_CUDA_FLAGS="-O3 -DHAVE_MPI --ptxas-options=-v" \
      # -DCMAKE_BUILD_TYPE="Release"    \
cmake -DCMAKE_CXX_COMPILER=mpic++     \
      -DCMAKE_C_COMPILER=mpicc        \
      -DCMAKE_Fortran_COMPILER=mpif90 \
      -DYAKL_ARCH="CUDA"              \
      -DYAKL_PROFILE="On"             \
      -DYAKL_CUDA_FLAGS="-O3 -DHAVE_MPI --ptxas-options=-v" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DMPI_INCLUDE_PATH="${MPI_INCLUDE_PATH}" \
      -DLDFLAGS="-lpnetcdf" \
      ..
