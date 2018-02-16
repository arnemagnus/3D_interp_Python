from distutils.core import setup
import sys

try:
    from Cython.Build import cythonize
except ImportError:
    print('A working Cython installation is required to run this install script!')
    sys.exit(1)

setup(name="cytrilinearperiodic",
      ext_modules=cythonize("src/cytrilinearperiodic.pyx"))