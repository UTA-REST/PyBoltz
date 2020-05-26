'''from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os
import  numpy
from Cython.Distutils import build_ext


# https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
ext_modules=[
    Extension("*", ["Townsend/CollisionFrequencyCalc/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/PulsedTownsend/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/TimeOfFlight/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/SteadyStateTownsend/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/Friedland/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/Monte/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*",["Monte/*.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["*.pyx"],include_dirs=[numpy.get_include(),'.'])
]
setup(name = 'PyBoltz',
      ext_modules = ext_modules,
      # Inject our custom trigger
      cmdclass={'build_ext': build_ext}
      )
setup(ext_modules=cythonize(ext_modules))
'''
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import os
from Cython.Build import cythonize

import Cython
import numpy
from io import open


def returnPyxFiles(path):
    l = []
    for i in os.listdir(path):
        if i.endswith(".pyx") or i.endswith(".c"):
            l.append(path+i)
    return l


def returnPxdFiles(path):
    l = []
    for i in os.listdir(path):
        if i.endswith(".pxd"):
            l.append(path+i)
    return l

extensions = [
    Extension("*", ["Townsend/CollisionFrequencyCalc/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/PulsedTownsend/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/TimeOfFlight/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/SteadyStateTownsend/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/Friedland/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*", ["Townsend/Monte/*.pyx"], include_dirs=[numpy.get_include(), '.']),
    Extension("*",["Monte/*.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["*.pyx"],include_dirs=[numpy.get_include(),'.'])
]
setup(ext_modules=cythonize(extensions))
