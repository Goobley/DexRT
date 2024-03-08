#!/bin/bash

mkdir -p /src/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /src/miniconda3/miniconda.sh
bash /src/miniconda3/miniconda.sh -b -u -p /src/miniconda3