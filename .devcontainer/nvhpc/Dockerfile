FROM nvcr.io/nvidia/nvhpc:24.1-devel-cuda12.3-ubuntu22.04

RUN apt update && apt install -y git curl libcurl4-openssl-dev

WORKDIR /src/magma
COPY magma/docker_setup.sh .
RUN ./docker_setup.sh "-DMAGMA_ENABLE_CUDA=ON -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++"

WORKDIR /src/hdf5
COPY hdf5/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/netcdf
COPY netcdf/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/pnetcdf
COPY pnetcdf/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/conda
COPY miniconda/docker_setup.sh .
RUN ./docker_setup.sh

COPY magma/magma.conf /etc/ld.so.conf.d
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
COPY miniconda/env_setup.sh .
RUN [ "/bin/bash", "env_setup.sh" ]
ENV HDF5_USE_FILE_LOCKING=FALSE
