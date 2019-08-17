# PyBoltz
This software package is a translation of the Fortran based Magboltz into Cython. This project was done to allow for more productive work to be done with magboltz.

## General information.

### About Magboltz.
The Magboltz program computes drift gas properties by "numerically integrating the Boltzmann transport equation"-- i.e., simulating an electron bouncing around inside a gas. By tracking how far the virtual electron propagates, the program can compute the drift velocity. By including a magnetic field, the program can also calculate the Lorentz angle. [Read more](http://cyclo.mit.edu/drift/www/aboutMagboltz.html).

### Why Cython?
Cython's static typing improves the speed of python code by about a hundred times. In other words, Cython provides us with the simplicity of python and the speed of Fortran/C. [Read more](https://cython.org/).

## Setting up and running instructions. 

### Setting up.
To be able to run this project you will need python3+, cython, and numpy installed. The setup that we use has python 3.6.7, Cython 0.29.3, and numpy 1.16.1. 

### Gases cross section database.
Before building the code make sure to run the following commands in the Cython directory to get the gases.npy file made, as this file has all the cross section values.
```
$ python3 Setup_npy.py
```

### Building.
To build the code clone this project and run the following command in the Cython directory. This should compile all of the Cython and add the path to your PYTHONPATH so you can access the libraries for anywhere. This will take a few minutes the first time.
```
$ source setup.sh
```

Please note that you might need to change the commands inside the setup.sh file to match your python version.

### Running PyBoltz.
To run the code, you will need to import Magboltz and instantiate an instance of the Magboltz object, fill in the input parameters and call the Magboltz.Start() function. There are examples in the Examples directory on to how to use PyBoltz. The main example is the Test_Magboltz.py code. This example also has a list of the gases in PyBoltz.

#### Input parameters.
* **Magboltz.NGAS** - The number of gases in the mixture (goes up to 6).
* **Magboltz.NMAX** - The number of simulated events / 2*10E7.
* **Magboltz.IPEN** - Penning effects included (0 or 1).
* **Magboltz.ITHRM** - Thermal motion included (0 or 1).
* **Magboltz.EFINAL** - Upper limit of electron energy integration (0.0 to automatically calculate this value).
* **Magboltz.NGASN** - Array of six elements that has the number of each gas in the mixture.
* **Magboltz.FRAC** - Array of six elements that has the percentage of each gas in the mixture.
* **Magboltz.TEMPC** - The tempreture in degrees centigrade.
* **Magboltz.TORR** - The pressire \[torr\].
* **Magboltz.EFIELD** - The electric field in the chamber \[Volts/Cm\].
* **Magboltz.BMAG** - The magnitude of the magentic field \[Tesla\].
* **Magboltz.BTHETA** - The angle between the magentic field and the electric field. 
* **Magboltz.NANISO** - This variable is used to fix the angular distrubtions to one of the following types. 
  - Okhrimvoskky Type - Magboltz.NANISO = 2 (default value).
  - Capitelli Longo Type - Magboltz.NANISO = 1.
  - Isotropic Scattering - Magboltz.NANISO = 0.
  
#### Output parameters.
Please note that the following are only the main output parameters. One can still get any value from the parameters within the Magboltz class.

* **Magboltz.WZ** - Drift velocity in the Z direction.
* **Magboltz.WY** - Drift velocity in the Y direction.
* **Magboltz.WX** - Drift velocity in the X direction.
* **Magboltz.DWZ** - Error for the Magboltz.WZ value (+- Magboltz.DWZ * Magboltz.WZ).
* **Magboltz.DWY** - Error for the Magboltz.WY value (+- Magboltz.DWY * Magboltz.WY).
* **Magboltz.DWX** - Error for the Magboltz.WX value (+- Magboltz.DWX * Magboltz.WX).
* **Magboltz.DIFTR** - Transverse diffusion.
* **Magboltz.DFTER** - Error for the Magboltz.DIFTR value (+- Magboltz.DFTER * Magboltz.DIFTR).
* **Magboltz.DIFLN** - Longitudinal diffusion.
* **Magboltz.DFLER** - Error for the Magboltz.DIFLN value (+- Magboltz.DFLER * Magboltz.DIFLN).
* **Magboltz.DTMN** - Transverse diffusion.
* **Magboltz.DFTER1** - Error for the Magboltz.DTMN value (+- Magboltz.DTMN * Magboltz.DFTER1).
* **Magboltz.DLMN** - Longitudinal diffusion.
* **Magboltz.DFLER1** - Error for the Magboltz.DLMN vlaue (+- Magboltz.DLMN * Magboltz.DFLER1).
* **Magboltz.AVE** - Mean electron energy.
* **Magboltz.DEN** - Error for the Magboltz.AVE value (+- Magboltz.AVE * Magboltz.DEN).
* **Magboltz.DIFXX** - Diffusion in the X plane.
* **Magboltz.DXXER** - Error for the Magboltz.DIFXX value (+- Magboltz.DIFXX * Magboltz.DXXER).
* **Magboltz.DIFYY** - Diffusion in the Y plane.
* **Magboltz.DYYER** - Error for the Magboltz.DIFYY value (+- Magboltz.DIFYY * Magboltz.DYYER).
* **Magboltz.DIFZZ** - Diffusion in the Z plane.
* **Magboltz.DZZER** - Error for the Magboltz.DIFZZ value (+- Magboltz.DIFZZ * Magboltz.DZZER).
* **Magboltz.DIFYZ** - Diffusion in the YZ plane.
* **Magboltz.DYZER** - Error for the Magboltz.DIFYZ value (+- Magboltz.DIFYZ * Magboltz.DYZER).
* **Magboltz.DIFXY** - Diffusion in the XY plane.
* **Magboltz.DXYER** - Error for the Magboltz.DIFXY value (+- Magboltz.DIFXY * Magboltz.DXYER).
* **Magboltz.DIFXZ** - Diffusion in the XZ plane.
* **Magboltz.DXZER** - Error for the Magboltz.DIFXZ value (+- Magboltz.DIFXZ * Magboltz.DXZER).

## Gas list.
The current PyBoltz version has the following gases. Please note that the number of the gas is used as an indicator to that gas in the code. 

* **CF4** - Gas # 1.
* **Argon** Gas # 2.
* **Helium-4** Gas # 3.
* **Helium-3** Gas # 4.
* **Neon** Gas # 5.
* **Krypton** Gas #6.
* **Xenon** Gas # 7.
* **CH4** Gas # 8.
* **Ethane** Gas # 9.
* **Propane** Gas # 10.
* **Isobutane** Gas # 11.
* **CO2** Gas # 12.
* **H2O** Gas # 14.
* **Oxygen** Gas # 15.
* **Nitrogen** Gas # 16.
* **Hydrogen** Gas # 21.
* **Deuterium** Gas # 22.
* **DME** Gas # 25.


[Documentaion...](https://uta-rest.github.io/PyBoltz-Documentation/html/).
