#!/bin/bash

NETCDF_VERSION="4.9.2"
rm -rf netcdf-c-${NETCDF_VERSION}*
HDF_DIR="/src/hdf5/install"

wget "https://downloads.unidata.ucar.edu/netcdf-c/${NETCDF_VERSION}/netcdf-c-${NETCDF_VERSION}.tar.gz"
tar xvzf "netcdf-c-${NETCDF_VERSION}.tar.gz"
cd "netcdf-c-${NETCDF_VERSION}"
./configure --disable-libxml2 CC=$(which mpicc) CPPFLAGS="-I${HDF_DIR}/include" LDFLAGS="-L${HDF_DIR}/lib"
make -j 10
make install