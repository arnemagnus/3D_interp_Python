from distutils.core import setup
from Cython.Build import cythonize

setup(name="shrubbery",ext_modules=cythonize("src/cytrilinear.pyx"))
