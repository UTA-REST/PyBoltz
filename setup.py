from setuptools import setup, Extension
from Cython.Distutils import build_ext
import glob
import os
import Cython
import numpy
from io import open

extensions = [
    Extension("PyBoltz.Townsend.CollisionFrequencyCalc", [glob.glob("PyBoltz/Townsend/CollisionFrequencyCalc/*.pyx"),"PyBoltz/Townsend/CollisionFrequencyCalc/*.pxd"], include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.PulsedTownsend", ["PyBoltz/Townsend/PulsedTownsend/*.pyx","PyBoltz/Townsend/PulsedTownsend/*.pxd"], include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.TimeOfFlight", ["PyBoltz/Townsend/TimeOfFlight/*.pyx","PyBoltz/Townsend/TimeOfFlight/*.pxd"], include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.SteadyStateTownsend", ["PyBoltz/Townsend/SteadyStateTownsend/*.pyx","PyBoltz/Townsend/SteadyStateTownsend/*.pxd"], include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.Friedland", ["PyBoltz/Townsend/Friedland/*.pyx","PyBoltz/Townsend/Friedland/*.pxd"], include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend", ["PyBoltz/Townsend/*.pyx","PyBoltz/Townsend/*.pxd"], include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.Monte", ["PyBoltz/Townsend/Monte/*.pyx","PyBoltz/Townsend/Monte/*.pxd"], include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Monte",["PyBoltz/Monte/*.pyx","PyBoltz/Monte/*.pxd"],include_dirs=[numpy.get_include(),'.']),
    Extension("PyBoltz",["PyBoltz/*.pyx","PyBoltz/*.pxd"],include_dirs=[numpy.get_include(),'.'])
]
setup(
    setup_requires=[
        'cython>=0.2',
    ],
    zip_safe=False,
    name='PyBoltz',  # Required
    packages=['PyBoltz'],

    version='1.1.0',  # Required
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    package_dir={'PyBoltz': 'PyBoltz'},

    install_requires=['numpy','cython','PyGasMix @ git+https://github.com/UTA-REST/PyGasMix.git#egg=PyGasMix-1.1.0'],  # Optional
    include_package_data = True,
    package_data={  # Optional 
        'PyBoltz': ['./PyBoltz/*.pxd'],
    },
    ext_modules = extensions,
    cmdclass={'build_ext': build_ext},
)
