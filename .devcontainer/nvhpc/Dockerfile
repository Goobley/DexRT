FROM nvcr.io/nvidia/nvhpc:24.5-devel-cuda12.4-ubuntu22.04 AS cuda-base

RUN apt update && apt install -y git curl libcurl4-openssl-dev

WORKDIR /src/magma
COPY docker_deps/magma/docker_setup.sh .
RUN ./docker_setup.sh "-DMAGMA_ENABLE_CUDA=ON -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++"

WORKDIR /src/hdf5
COPY docker_deps/hdf5/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/netcdf
COPY docker_deps/netcdf/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/pnetcdf
COPY docker_deps/pnetcdf/docker_setup.sh .
RUN ./docker_setup.sh

# WORKDIR /src/kokkos
# COPY docker_deps/kokkos/docker_setup.sh .
# RUN ./docker_setup.sh "-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON -DKokkos_ENABLE_THREADS=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_DEBUG=ON"

WORKDIR /src/conda
COPY docker_deps/miniconda/docker_setup.sh .
RUN ./docker_setup.sh

COPY docker_deps/magma/magma.conf /etc/ld.so.conf.d
RUN ldconfig

# NOTE(cmo): Based on https://github.com/adalundhe/dev-images/blob/main/cpp/Dockerfile.cpp
ARG USERNAME=devcontainer
ARG USER_UID=5000
ARG USER_GID=$USER_UID

RUN apt install -y sudo

RUN groupadd --gid $USER_GID $USERNAME \
    && adduser --disabled-password --gecos "" --uid $USER_UID --gid $USER_GID $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ENV HOME=/home/$USERNAME

USER $USERNAME

WORKDIR /src/conda
COPY docker_deps/miniconda/env_setup.sh .
RUN [ "/bin/bash", "env_setup.sh" ]
ENV HDF5_USE_FILE_LOCKING=FALSE
