#!/bin/bash

# Call with $1 as a string of CMake flags e.g. -DKOKKOS_ENABLE_CUDA=ON
wget https://github.com/kokkos/kokkos/releases/download/4.4.01/kokkos-4.4.01.tar.gz
tar xvzf kokkos-4.4.01.tar.gz
cd kokkos-4.4.01
mkdir build
cd build
cmake $1 ..
make -j 24
make install