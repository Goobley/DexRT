FROM ubuntu:latest

# NOTE(cmo): Based on https://github.com/adalundhe/dev-images/blob/main/cpp/Dockerfile.cpp
ARG USERNAME=devcontainer
ARG USER_UID=5000
ARG USER_GID=$USER_UID

RUN apt update && apt install -y build-essential curl libcurl4-openssl-dev libopenmpi-dev cmake wget git libopenblas-dev liblapack-dev sudo

WORKDIR /src/hdf5
COPY docker_deps/hdf5/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/netcdf
COPY docker_deps/netcdf/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/pnetcdf
COPY docker_deps/pnetcdf/docker_setup.sh .
RUN ./docker_setup.sh

WORKDIR /src/conda
COPY docker_deps/miniconda/docker_setup.sh .
RUN ./docker_setup.sh

RUN ldconfig

RUN groupadd --gid $USER_GID $USERNAME \
    && adduser --uid $USER_UID --gid $USER_GID $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ENV HOME=/home/$USERNAME

USER $USERNAME

WORKDIR /src/conda
COPY docker_deps/miniconda/env_setup.sh .
RUN [ "/bin/bash", "env_setup.sh" ]
ENV HDF5_USE_FILE_LOCKING=FALSE
