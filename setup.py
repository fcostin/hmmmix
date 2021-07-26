from setuptools import find_packages, setup
from Cython.Build import cythonize

setup(
    packages = find_packages("lib"),
    package_dir = {"": "lib"},
    package_data = {},
    ext_modules = cythonize(
        "lib/hmmmix/trellis/libtrellis.pyx",

        # Generate a an annotated report of the cython code highlighting
        # points where interactions with the python interpreter occur.
        # The report will be written to lib/hmmmix/trellis/libtrellis.html .
        annotate=True,

        compiler_directives={
            'language_level' : "3", # Py3.
        },
    )
)
