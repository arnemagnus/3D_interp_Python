from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension('cytrilinearnonperiodic', ['src/cytrilinearnonperiodic.pyx'],
                  include_dirs = [numpy.get_include(),
                                  './include']),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension('cytrilinearnonperiodic', ['src/cytrilinearnonperiodic.c'],
                  include_dirs = [numpy.get_include(),
                                  './include']),
    ]

setup(
    name = 'Eels',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
)

