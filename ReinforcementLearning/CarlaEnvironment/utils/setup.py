from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = Extension("dist_c", ["dist_c.pyx"])
setup(

    ext_modules = cythonize(
        extensions,
        compiler_directives = {'language_level':"3"}
        )
)