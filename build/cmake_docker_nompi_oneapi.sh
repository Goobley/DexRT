#!/bin/bash
GCC_INCLUDE_PATH=$(icpx --print-file-name=include)

./cmake_clean.sh

      # -DYAKL_CUDA_FLAGS="-O3 --ptxas-options=-v" \
      # -DYAKL_DEBUG="On"               \
      # -DYAKL_B4B="On"                  \
      # -DYAKL_SYCL_FLAGS="-fsycl -g -O0 -std=c++20 -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -Wdouble-promotion -Wimplicit-float-conversion" \
      # -DYAKL_SYCL_FLAGS="-fsycl -O3 -std=gnu++20 -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -Xs \"-device adlp\"" \
cmake -DCMAKE_CXX_COMPILER=icpx     \
      -DCMAKE_C_COMPILER=icx        \
      -DCMAKE_Fortran_COMPILER=ifx \
      -DYAKL_ARCH="SYCL"              \
      -DYAKL_PROFILE="On"             \
      -DYAKL_SYCL_FLAGS="-fsycl -O3 -std=gnu++20 -sycl-std=2020 -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel" \
      -DYAKL_MANAGED_MEMORY="On"           \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs)" \
      ..
