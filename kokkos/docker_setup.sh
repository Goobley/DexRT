#!/bin/bash

KOKKOS_VERSION="4.3.00"

wget "https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.tar.gz"
tar xvzf "${KOKKOS_VERSION}.tar.gz"
cd kokkos-${KOKKOS_VERSION}
mkdir build
cd build
#  cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=ON ..
cmake $1 ..
make -j 20
make install