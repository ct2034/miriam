from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="odrm",
    # ext_modules=cythonize("odrm.pyx"),
    version="1.0.0",
    packages=["odrm"],
    package_dir={"": "src"},
)
