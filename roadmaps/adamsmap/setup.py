from distutils.core import setup
from Cython.Build import cythonize

setup(name='cadamsmap',
      # ext_modules=cythonize("adamsmap.pyx"),
      version='1.0.0',
      packages=['adamsmap'],
      package_dir={'': 'src'}
      )
