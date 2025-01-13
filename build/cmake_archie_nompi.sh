#!/bin/bash

source setup_archie_cuda_env.sh
GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MAGMA_BASE_PATH="/opt/software/magma/gcc-8.5.0/2.8.0/Xeon-5218R"
MAGMA_INCLUDE_PATH="${MAGMA_BASE_PATH}/include/"
MAGMA_LIB_PATH="${MAGMA_BASE_PATH}/lib/"
MAGMA_LIB="-lmagma"
CUDA_MATH_INCLUDE_PATH="/opt/software/nvidia/sdk/Linux_x86_64/24.5/math_libs/12.4/"
CUDA_MATH_LIB_PATH="/opt/software/nvidia/sdk/Linux_x86_64/24.5/math_libs/12.4/lib64/"

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_CUDA_COMPILER=nvcc    \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="CUDA"              \
      -DYAKL_AUTO_PROFILE="On"             \
      -DYAKL_CUDA_FLAGS="-O3 --generate-line-info" \
      -DYAKL_INT64_RESHAPE="On"       \
      -DDEXRT_CUDA_ARCHITECTURES="80" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DMPI_INCLUDE_PATH="" \
      -DCUDA_MATH_INCLUDE_PATH="${CUDA_MATH_INCLUDE_PATH}" \
      -DMAGMA_INCLUDE_PATH="${MAGMA_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="-L${CUDA_MATH_LIB_PATH} $(nc-config --libs) -L${MAGMA_LIB_PATH} ${MAGMA_LIB}" \
      -DLDFLAGS="-Wl,-rpath,${MAGMA_LIB_PATH}:$(nc-config --libdir):${CUDA_MATH_LIB_PATH}" \
      ..
