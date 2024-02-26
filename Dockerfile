FROM nvcr.io/nvidia/nvhpc:24.1-devel-cuda12.3-ubuntu22.04

WORKDIR /src/pnetcdf
COPY pnetcdf/docker_setup.sh .
RUN ./docker_setup.sh

RUN addgroup --gid 1000 devcontainer
RUN adduser --disabled-password --uid 1000 --gid 1000 devcontainer
ENV HOME /home/devcontainer

USER devcontainer

