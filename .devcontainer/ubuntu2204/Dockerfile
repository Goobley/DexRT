FROM ubuntu:latest

RUN apt update && apt install -y build-essential curl libcurl4-openssl-dev libopenmpi-dev cmake wget git

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

RUN ldconfig

RUN addgroup --gid 1000 devcontainer
RUN adduser --disabled-password --uid 1000 --gid 1000 devcontainer
ENV HOME /home/devcontainer

USER devcontainer

WORKDIR /src/conda
COPY miniconda/env_setup.sh .
RUN [ "/bin/bash", "env_setup.sh" ]
ENV HDF5_USE_FILE_LOCKING=FALSE
