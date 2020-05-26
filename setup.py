from setuptools import setup, Extension
from Cython.Distutils import build_ext

import os
import Cython
import numpy
from io import open

extensions = [
        Extension("PyBoltz.src.PyBoltz",["PyBoltz/src/PyBoltz.pyx","PyBoltz/src/PyBoltz.pxd"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyBoltz/']),

'''
    Extension("PyGasMix.Gases.GasUtil",["PyGasMix/Gases/GasUtil.pyx","PyGasMix/Gases/GasUtil.pxd"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gasmix",["PyGasMix/Gasmix.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.ARGON",["PyGasMix/Gases/ARGON.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.CF4",["PyGasMix/Gases/CF4.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.CH4",["PyGasMix/Gases/CH4.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.CO2",["PyGasMix/Gases/CO2.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.DEUTERIUM",["PyGasMix/Gases/DEUTERIUM.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.DME",["PyGasMix/Gases/DME.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.ETHANE",["PyGasMix/Gases/ETHANE.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.H2O",["PyGasMix/Gases/H2O.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.HELIUM3",["PyGasMix/Gases/HELIUM3.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.HELIUM4",["PyGasMix/Gases/HELIUM4.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.HYDROGEN",["PyGasMix/Gases/HYDROGEN.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.ISOBUTANE",["PyGasMix/Gases/ISOBUTANE.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.KRYPTON",["PyGasMix/Gases/KRYPTON.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.NEON",["PyGasMix/Gases/NEON.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.NITROGEN",["PyGasMix/Gases/NITROGEN.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.OXYGEN",["PyGasMix/Gases/OXYGEN.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.PROPANE",["PyGasMix/Gases/PROPANE.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.XENON",["PyGasMix/Gases/XENON.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
    Extension("PyGasMix.Gases.XENONMERT",["PyGasMix/Gases/XENONMERT.pyx"],include_dirs=[numpy.get_include(),os.getcwd()+'/PyGasMix/']),
'''
]
setup(
    setup_requires=[
        'cython>=0.2',
    ],
    zip_safe=False,
    name='PyBoltz',  # Required
    packages=['PyBoltz'],

    version='1.0.0',  # Required
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    package_dir={'PyBoltz': 'PyBoltz'},

    install_requires=['numpy','cython','-e git+https://github.com/UTA-REST/PyGasMix.git#egg=PyGasMix'],  # Optional
    #include_package_data = True,
    #'''package_data={  # Optional
    #    'PyGasMix': ['./PyGasMix/*.pxd','./PyGasMix/*.pxd'],
    #    'PyGasMix/Gases': ['./PyGasMix/Gases/*.pxd','./PyGasMix/Gases/gases.npy'],
    #},
    #ext_modules = extensions,
    #cmdclass={'build_ext': build_ext},
)
