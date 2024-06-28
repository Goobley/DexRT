#!/bin/bash

./cmake_clean.sh

      # -DYAKL_PROFILE="On"             \
cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="OPENMP"              \
      -DYAKL_OPENMP_FLAGS="-O3 -fopenmp -fopenmp-simd" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs) -fopenmp" \
      ..
