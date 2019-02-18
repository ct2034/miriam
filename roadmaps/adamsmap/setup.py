from distutils.core import setup
from Cython.Build import cythonize

setup(name='cadamsmap',
      ext_modules=cythonize("adamsmap.pyx"))
