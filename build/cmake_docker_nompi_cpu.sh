#!/bin/bash
GCC_INCLUDE_PATH=$(g++ --print-file-name=include)

./cmake_clean.sh

      # -DYAKL_PROFILE="On"             \
      # -DYAKL_VERBOSE="On"               \
cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="OPENMP"              \
      -DYAKL_OPENMP_FLAGS="-O3 -fopenmp -fopenmp-simd" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs) -fopenmp" \
      ..
