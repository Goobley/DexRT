#!/bin/bash

./cmake_clean.sh

      # -DYAKL_PROFILE="On"             \
cmake -DCMAKE_CXX_COMPILER=g++     \
      -DCMAKE_C_COMPILER=gcc        \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DYAKL_ARCH=""              \
      -DYAKL_CXX_FLAGS="-O3" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs)" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DDEX_FORCE_DOUBLE_PREC=Off \
      -DDEX_HALF_RADIATION_STORAGE=On \
      ..
