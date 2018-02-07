from distutils.core import setup, Extension

module1 = Extension('demo',sources=['src/cytricubic.c'])

setup (name = 'shrubbery',
        version = '1.0',
        description = 'Test',
        ext_modules=[module1])
