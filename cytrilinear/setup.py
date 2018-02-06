from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
from Cython.Build import cythonize

#ext_modules=[Extension("shrubbery",["cy_shrub.pyx"])]

#setup(name="MyProject",cmdclass={"build_ext":build_ext},ext_modules=ext_modules)

setup(name="shrubbery",ext_modules =cythonize("src/cytrilinear.pyx"))
