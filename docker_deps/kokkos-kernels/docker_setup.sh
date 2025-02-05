#!/bin/bash

# Call with $1 as a string of CMake flags e.g. -DKOKKOS_ENABLE_CUDA=ON
wget -O kokkos-kernels.tar.gz https://github.com/kokkos/kokkos-kernels/releases/download/4.5.01/kokkos-kernels-4.5.01.tar.gz
tar xvzf kokkos-kernels.tar.gz
cd kokkos-kernels-4.5.01
mkdir build
cd build
cmake $1 ..
make -j 24
make install