from distutils.core import setup, Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("Shrubbery", ["src/cytrilinear.pyx"]),
        Extension("Hovercraft", ["src/cytricubic.pyx"])]

setup(
        name = "Snake",
        cmdclass = {"build_ext": build_ext},
        ext_modules = ext_modules
)

#setup(name="shrubbery",ext_modules=cythonize("src/cytrilinear.pyx"))
