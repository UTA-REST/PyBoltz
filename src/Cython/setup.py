from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import  numpy
print(numpy.get_include())
ext_modules=[
    Extension("*",["Gases/*.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["Monte/*.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["*.pyx"],include_dirs=[numpy.get_include(),'.'])
]
for e in ext_modules:
    e.cython_directives = {"embedsignature": True}
setup(ext_modules=cythonize(ext_modules))
