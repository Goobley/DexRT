#!/bin/bash

. /local/scratch/osborne/spack/share/spack/setup-env.sh

spack load cmake
spack load magma
spack load netcdf-c
spack load nvhpc

