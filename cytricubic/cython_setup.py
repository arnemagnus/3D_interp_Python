from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

#extensions = [
#        Extension('shrubbery', ['src/cytricubic.pyx'],
#                    include_dirs=[np.get_include(), './include'])
#        ]

#setup(
#        ext_modules = cythonize(extensions)
#        )

setup(name="shrubbery",ext_modules =cythonize("src/cytricubic.pyx"))
