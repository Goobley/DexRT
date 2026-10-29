#!/bin/bash


./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH="OPENMP"              \
      -DYAKL_OPENMP_FLAGS="-O2 -fopenmp" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDFLAGS="$(nc-config --libs)" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DDEX_FORCE_DOUBLE_PREC=Off \
      -DDEX_HALF_RADIATION_STORAGE=On \
      ..