
# setup the enviorment                                                                             
export PYTHONPATH=$PYTHONPATH:$PWD
python3 Setup_npy.py

python3 setup.py clean

python3 setup.py build_ext --inplace
