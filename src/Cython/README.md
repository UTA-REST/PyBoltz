# Cython directory

In this directory exists the PyBoltz object, Gas object, Gasmix object, Ang object, Energylimits module, Setups module, Setup_npy module, Mixers module, PyBoltzRun module, Gases module, the Monte module.

## PyBoltz object

This is the main object, hence the name PyBoltz. In this object all the input, output, and intermediate parameters are stored. This object's functions coordinate the correct use of the modules. To eleaborate, after setting up the PyBoltz with the input parameters, the PyBoltz object calls the functions that correspond to the given input.

## Ang object

This object is used to set angle cuts on angular distribution and renormalise forward scattering probability.

## Energylimits modules

This module has the energy limits functions for the different input parameters.

## Setups module

This module has a single function that is used to setup the constants needed for the simulation.

## Mixers module 
This module has the Mixing functions. In those functions the Gasmix object is used to run the gas functions. After doing so the Mixing functions use the output to calculate the collision frequencies that are needed for the simulation.

## PyBoltzRun
This module is a wrapper build to ease the use of PyBoltz. Check the Examples/Example.py file for a usage example.

## OdieRun
This module provides functionality to run PyBoltz similar to PyBoltzRun and includes the ability to output gas files in the Garfield++ format. It should be able to be used as a drop-in replacement where PyBoltzRun is used. Check Examples/Example_Odie.py for an example on how to use it.

## Monte module 
This module has all the Monte carlo functions used in PyBoltz.
