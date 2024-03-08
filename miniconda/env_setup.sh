#!/bin/bash

/src/miniconda3/bin/conda init bash
/src/miniconda3/bin/conda create -n py312 python=3.12
source /src/miniconda3/bin/activate py312
python -m pip install numpy scipy matplotlib netcdf4 lightweaver
