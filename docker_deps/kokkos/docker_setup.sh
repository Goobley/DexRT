#!/bin/bash

# Call with $1 as a string of CMake flags e.g. -DKOKKOS_ENABLE_CUDA=ON
wget -O kokkos.tar.gz https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
tar xvzf kokkos.tar.gz
cd kokkos-4.5.01
mkdir build
cd build
cmake $1 ..
make -j 24
make install