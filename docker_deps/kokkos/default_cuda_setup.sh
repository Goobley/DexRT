#!/bin/bash
rm -r kokkos-4.5.01

./docker_setup.sh \
    "-DKokkos_ENABLE_SERIAL=ON \
     -DKokkos_ENABLE_OPENMP=ON \
     -DKokkos_ENABLE_CUDA=ON \
     -DKokkos_ARCH_ADA89=ON \
     -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
     -DKokkos_ENABLE_CUDA_LAMBDA=ON \
     -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON \
     -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=$(pwd)/../../kokkos"