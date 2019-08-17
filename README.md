# PyBoltz
This software package is a translation of the Fortran based Magboltz into Cython. This project was done to allow for more productive work to be done with magboltz.

## General information
### About Magboltz
The Magboltz program computes drift gas properties by "numerically integrating the Boltzmann transport equation"-- i.e., simulating an electron bouncing around inside a gas. By tracking how far the virtual electron propagates, the program can compute the drift velocity. By including a magnetic field, the program can also calculate the Lorentz angle. [Read more](http://cyclo.mit.edu/drift/www/aboutMagboltz.html).

### Why Cython?
Cython's static typing improves the speed of python code by about a hundred times. In other words, Cython provides us with the simplicity of python and the speed of Fortran/C. [Read more](https://cython.org/).


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
