# setup the enviorment
export PYTHONPATH=$PYTHONPATH:$PWD

# setup the cross sections database (gases.npy)
python3 Setup_npy.py

# build the code
python3 setup.py clean
export CFLAGS="-I /usr/local/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
python3 setup.py build_ext --inplace
