#!/bin/bash

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="OPENMP"              \
      -DYAKL_PROFILE="On"             \
      -DYAKL_OPENMP_FLAGS="-O3 -fopenmp" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDFLAGS="$(nc-config --libs)" \
      ..
