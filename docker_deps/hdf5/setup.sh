#!/bin/bash

HDF_VERSION="1_14_3"
SZIP_VERSION="2.1.1"
rm -rf "szip-${SZIP_VERSION}*"
rm -rf "hdf5-${HDF_VERSION}*"
rm -rf hdfsrc

wget "https://github.com/HDFGroup/hdf5/releases/download/hdf5-${HDF_VERSION}/hdf5-${HDF_VERSION}.tar.gz"
tar xvzf "hdf5-${HDF_VERSION}.tar.gz"
cd hdfsrc
./configure CC=$(which mpicc) --enable-parallel --with-default-api-version=v110 --prefix="$(pwd)/../install"
make -j 10
make install