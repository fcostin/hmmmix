from setuptools import find_packages, setup
from Cython.Build import cythonize

setup(
    packages = find_packages("lib"),
    package_dir = {"": "lib"},
    package_data = {},
    ext_modules = cythonize("lib/hmmmix/trellis/libtrellis.pyx")
)
