# Cython/Gases directory 
This directory contains all the gas functions that PyBoltz has. Each gas function takes a pointer to a Gas struct object to fill that object with the outputs. 

Nonetheless, the gas functions take all of their cross section data from the gases.npy file in the Cython directory. Then the gas function runs through 4000 energy steps, where it interpolates the data in the cross section with each energy step. Those outputs are stored in arrays that are then used in the Mixer functions to calculate the collision frequencies. 

