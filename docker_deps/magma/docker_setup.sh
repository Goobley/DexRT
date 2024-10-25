#!/bin/bash

# Call with $1 as a string of CMake flags e.g. -DMAGMA_ENABLE_CUDA=ON

wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.8.0.tar.gz
tar xvzf magma-2.8.0.tar.gz
cd magma-2.8.0
mkdir build
cd build
cmake $1 ..
make -j 24
make install