#!/usr/bin/env bash
# setup the enviorment
temp="/PyBoltz/"
export PYTHONPATH=$PYTHONPATH:$PWD$temp
export PATH=$PATH:$PWD$temp
echo $PYTHONPATH

# build the code
python3 setup_build.py clean
export CFLAGS="-I /usr/local/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
python3 setup_build.py build_ext --inplace


