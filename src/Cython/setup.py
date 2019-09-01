from distutils.core import setup
from Cython.Build import cythonize
import  numpy
print(numpy.get_include())

setup(ext_modules=cythonize(["gases/*.pyx"]), include_dirs=[numpy.get_include(),'.'])
setup(ext_modules=cythonize(["Monte/*.pyx"]), include_dirs=[numpy.get_include(),'.'])
#setup(ext_modules=cythonize(["Misc/*.pyx"]), include_dirs=[numpy.get_include(),'.'])
setup(ext_modules=cythonize(["*.pyx"]), include_dirs=[numpy.get_include(),'.'])
