import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

print(numpy.get_include())


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    home = '/usr/local/cuda-10.1/'
    nvcc = pjoin(home, 'bin', 'nvcc')

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



CUDA = locate_cuda()

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext = Extension('PyBoltz_Gpu',
        sources = ['MonteTGpu.cu','MonteGpu.cu', 'PyBoltz_Gpu.pyx'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args= {
            'gcc': [],
            'nvcc': [
                '-Xcompiler', '-fPIC', '-c', '-O2'
                ]
            },
            include_dirs = [numpy_include, CUDA['include'], 'src']
        )



setup(name = 'PyBoltz_Gpu',

      ext_modules = [ext],

      # Inject our custom trigger
      cmdclass = {'build_ext': custom_build_ext},

      # Since the package has c code, the egg cannot be zipped
      zip_safe = False)
ext_modules=[
    Extension("*",["Gases/*.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["Monte/*.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["Ang.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["EnergyLimits.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["Gasmix.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["MBSorts.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["Mixers.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["Setups.pyx"],include_dirs=[numpy.get_include(),'.']),
Extension("*",["PyBoltz.pyx"],include_dirs=[numpy.get_include(),'.'])
]
setup(ext_modules=cythonize(ext_modules))

