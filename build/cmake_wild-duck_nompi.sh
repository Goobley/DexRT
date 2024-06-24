#!/bin/bash

source setup_wild-duck_env.sh
GCC_INCLUDE_PATH=$(g++ --print-file-name=include)
MAGMA_BASE_PATH=$(spack location -i magma)
MAGMA_INCLUDE_PATH="${MAGMA_BASE_PATH}/include/"
MAGMA_LIB_PATH="${MAGMA_BASE_PATH}/lib/"
MAGMA_LIB="-lmagma"

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_CUDA_COMPILER=nvcc    \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="CUDA"              \
      -DYAKL_PROFILE="On"             \
      -DYAKL_CUDA_FLAGS="-O3 --generate-line-info" \
      -DYAKL_INT64_RESHAPE="On"       \
      -DDEXRT_CUDA_ARCHITECTURES="86" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DMPI_INCLUDE_PATH="" \
      -DMAGMA_INCLUDE_PATH="${MAGMA_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs) -L${MAGMA_LIB_PATH} ${MAGMA_LIB}" \
      -DLDFLAGS="-Wl,-rpath,${MAGMA_LIB_PATH}:$(nc-config --libdir)" \
      ..
