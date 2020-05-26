from setuptools import setup, Extension
from Cython.Distutils import build_ext
import os
import Cython
import numpy
from io import open


def returnPyxFiles(path):
    l = []
    for i in os.listdir(path):
        if i.endswith(".pyx") or i.endswith(".pxd"):
            l.append(path+i)
    return l
extensions = [
    Extension("PyBoltz.Townsend.CollisionFrequencyCalc", returnPyxFiles("PyBoltz/Townsend/CollisionFrequencyCalc/"), include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.PulsedTownsend", returnPyxFiles("PyBoltz/Townsend/PulsedTownsend/"), include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.TimeOfFlight", returnPyxFiles("PyBoltz/Townsend/TimeOfFlight/"), include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.SteadyStateTownsend", returnPyxFiles("PyBoltz/Townsend/SteadyStateTownsend/"), include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.Friedland", returnPyxFiles("PyBoltz/Townsend/Friedland/"), include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend", returnPyxFiles("PyBoltz/Townsend/"), include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Townsend.Monte", returnPyxFiles("PyBoltz/Townsend/Monte/"), include_dirs=[numpy.get_include(), '.']),
    Extension("PyBoltz.Monte",returnPyxFiles("PyBoltz/Monte/"),include_dirs=[numpy.get_include(),'.']),
    Extension("PyBoltz",returnPyxFiles("PyBoltz/"),include_dirs=[numpy.get_include(),'.'])
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
