
# setup the enviorment                                                                             
export PYTHONPATH=$PYTHONPATH:$PWD
echo $PYTHONPATH
python3 setup.py clean

python3 setup.py build_ext --inplace
