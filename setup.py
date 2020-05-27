from setuptools import setup, Extension
from Cython.Distutils import build_ext
import os
import Cython
import numpy
from io import open
import glob

def returnPyxFiles(path):
    l = []
    for i in os.listdir(path):
        if i.endswith(".pyx") or i.endswith(".c") or i.endswith(".pxd"):
            l.append(path+i)
    return l


def returnPxdFiles(path):
    l = []
    for i in os.listdir(path):
        if i.endswith(".pxd"):
            l.append(path+i)
    return l

def makeExtensions(path):
    extensions = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith(".pyx"):
                moduleFiles = [] 
                moduleFiles.append(root+'/'+file)
                if root+'/'+(file.split('.')[0]+'.pxd') in glob.glob(root+"/*.pxd"):
                    moduleFiles.append(root+'/'+(file.split('.')[0]+'.pxd'))
                pathWithFile = root+'/'+file.split('.')[0]
                moduleName = pathWithFile.replace('/','.')
                extensions.append(Extension(moduleName,moduleFiles,include_dirs=[numpy.get_include(),os.getcwd()+'/PyBoltz/',os.getcwd()+'/PyBoltz/C/']))
    return extensions
extensions = makeExtensions('PyBoltz_P')
'''[
    Extension("PyBoltz",returnPyxFiles("PyBoltz/"),include_dirs=[numpy.get_include(),os.getcwd()+'/PyBoltz/',os.getcwd()+'/PyBoltz/C/']),
    Extension("PyBoltz.Townsend.Monte", returnPyxFiles("PyBoltz/Townsend/Monte/")+returnPxdFiles("PyBoltz/Townsend/Monte/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/',os.getcwd()+'/PyBoltz/Townsend/Monte/']),
    Extension("PyBoltz.Townsend", returnPyxFiles("PyBoltz/Townsend/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.CollisionFrequencyCalc", returnPyxFiles("PyBoltz/Townsend/CollisionFrequencyCalc/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.PulsedTownsend", returnPyxFiles("PyBoltz/Townsend/PulsedTownsend/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.TimeOfFlight", returnPyxFiles("PyBoltz/Townsend/TimeOfFlight/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.SteadyStateTownsend", returnPyxFiles("PyBoltz/Townsend/SteadyStateTownsend/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.Friedland", returnPyxFiles("PyBoltz/Townsend/Friedland/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Monte",returnPyxFiles("PyBoltz/Monte/"),include_dirs=[numpy.get_include(),os.getcwd()+'/PyBoltz/']),
]'''
#print(returnPyxFiles("PyBoltz/Townsend/Monte/")+returnPxdFiles("PyBoltz/Townsend/Monte/"))
setup(
    setup_requires=[
        'cython>=0.2',
    ],
    zip_safe=False,
    name='PyBoltz_P',  # Required
    packages=['PyBoltz_P'],

    version='1.1.0',  # Required
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    package_dir={'PyBoltz_P': 'PyBoltz_P'},
    package_data={  # Optional
        'PyBoltz_P': returnPxdFiles("./PyBoltz_P/"),
        'PyBoltz_P/Monte': returnPxdFiles("PyBoltz_P/Monte/"),
        'PyBoltz_P/Townsend/Monte': returnPxdFiles("PyBoltz_P/Townsend/Monte/"),
        'PyBoltz_P/Townsend/Friedland': returnPxdFiles("PyBoltz_P/Townsend/Friedland/"),
        'PyBoltz_P/Townsend/SteadyStateTownsend': returnPxdFiles("PyBoltz_P/Townsend/SteadyStateTownsend/"),
        'PyBoltz_P/Townsend/TimeOfFlight': returnPxdFiles("PyBoltz_P/Townsend/TimeOfFlight/"),
        'PyBoltz_P/Townsend/PulsedTownsend': returnPxdFiles("PyBoltz_P/Townsend/PulsedTownsend/"),
        'PyBoltz_P/Townsend/CollisionFrequencyCalc': returnPxdFiles("PyBoltz_P/Townsend/CollisionFrequencyCalc/"),

    },

    install_requires=['numpy','cython','PyGasMix @ git+https://github.com/UTA-REST/PyGasMix.git#egg=PyGasMix-1.1.0'],  # Optional
    include_package_data = True,
    ext_modules = extensions,
    cmdclass={'build_ext': build_ext},
)

