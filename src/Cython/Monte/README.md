# Cython/Monte directory

This directory contains all the monte carlo functions in PyBoltz. Each one of these Monte carlo functions does the following. 

## Get a collision

In this part the simulation generates a collision by a monte carlo simulation that takes in mind the probabilities that come from the collision frequencies. This part spits out the collision time, velocity after the collision, and energy of the collision. 

## Process a collision

In this part the simulation gets the data of the collision and keeps track of the simulated electron's position, velocity, and energy. 

# Notes. 
PyBoltz runs a single elctron for a huge amount of collisions. However, to get the results for the diffusion numbers, PyBoltz decorrelates the same electron after N collisions. This is the same process as in Magboltz.

[Documentation...] (https://uta-rest.github.io/PyBoltz-Documentation/html/Monte.html).
