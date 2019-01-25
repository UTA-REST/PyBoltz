from distutils.core import setup
from Cython.Build import cythonize
import  numpy
setup(ext_modules=cythonize(["*.pyx"]), inlude_dirs=[numpy.get_include(),'.'])