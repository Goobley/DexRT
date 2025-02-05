#!/bin/bash
rm -r kokkos-kernels-4.5.01

    #  -DKokkosKernels_ENABLE_TPL_MAGMA=ON \
    #  -DKokkosKernels_MAGMA_ROOT=/usr/local/magma/ \
./docker_setup.sh \
    "-DCMAKE_BUILD_TYPE=Release \
     -DKokkosKernels_ENABLE_ALL_COMPONENTS=OFF \
     -DKokkosKernels_ENABLE_COMPONENT_BATCHED=ON \
     -DKokkosKernels_ENABLE_COMPONENT_BLAS=ON \
     -DKokkos_ROOT=$(pwd)/../../kokkos \
     -DCMAKE_PREFIX_PATH=$(pwd)/../../kokkos/lib/cmake \
     -DCMAKE_INSTALL_PREFIX=$(pwd)/../../kokkos-kernels-debug"