# MAGBOLTZ-py
This software package is a translation of the Fortran based Magboltz.
It is optimized to un using Cython and you will need python and cython
installed in order for it to compule and run.
during our development we used the following versions

Python 3.7.2

Cython version 0.29.6

to compile and setup the enviorment go to the directory 

MAGBOLTZ-py/src/Scripts/Cython

and run 

source setup.sh

This should compile all of the Cython and add the path to your PYTHONPATH so you can access the libaries for anywhere. 
This will take a few minuets the first time.
Once compiled run the template Test-Magboltz.py

python Test-Magboltz.py


This template includes the gas list and current output paramaters.
