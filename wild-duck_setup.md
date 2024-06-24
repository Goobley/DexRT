For a serial build
```
spack install nvhpc default_cuda=12.4
spack install cuda@12.4
spack install magma+cuda cuda_arch=80
spack install netcdf-c~mpi^hdf5~mpi
```

To build dex need to load modules

```
spack load nvhpc default_cuda=12.4
spack load magma
spack load netcdf-c
```
