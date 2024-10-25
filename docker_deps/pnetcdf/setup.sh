#!/bin/bash

PNETCDF_VERSION="1.12.3"
rm -rf pnetcdf-${PNETCDF_VERSION}*

wget https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz
tar xvzf "pnetcdf-${PNETCDF_VERSION}.tar.gz"
cd "pnetcdf-${PNETCDF_VERSION}"
./configure --prefix="$(pwd)/../install" MPICC="$(which mpicc)" MPICXX="$(which mpicxx)"
make -j 10
make install

