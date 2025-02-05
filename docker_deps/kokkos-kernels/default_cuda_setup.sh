#!/bin/bash
rm -r kokkos-kernels-4.5.01

    #  -DKokkosKernels_ENABLE_TPL_MAGMA=ON \
    #  -DKokkosKernels_MAGMA_ROOT=/usr/local/magma/ \
./docker_setup.sh \
    "-DCMAKE_BUILD_TYPE=Release \
     -DKokkos_ROOT=$(pwd)/../../kokkos \
     -DCMAKE_PREFIX_PATH=$(pwd)/../../kokkos/lib/cmake \
     -DCMAKE_INSTALL_PREFIX=$(pwd)/../../kokkos-kernels"