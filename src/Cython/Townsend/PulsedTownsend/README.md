# PT - Pulsed Townsend.
TOF - Time of flight
This directory has the PT.pyx. This file takes the output values in the TPlanes arraies. Those values are simulated by the Monte/MONTEFDT.pyx file. From the time, energy, velocity and count of the electrons at each of the 7 time planes it outputs values for each planes. Those values are used in the TimeOfFlight/TOF.pyx file. They include frequency of ionisation and attachment at each plane, velocity and energy.
