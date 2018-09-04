from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import numpy

import mpi4py
import os

class build_ext_subclass(build_ext):
    user_options = build_ext.user_options + \
            [
            ('mpicc', None, 'MPICC')
            ]
    def initialize_options(self):
        try:
            compiler = str(mpi4py.get_config()['mpicc'])
        except:
            compiler = "mpicc"

        self.mpicc = os.environ.get('MPICC', compiler)

        build_ext.initialize_options(self)

    def finalize_options(self):
        build_ext.finalize_options(self)

    def build_extensions(self):
        # turns out set_executables only works for linker_so, but for compiler_so
        self.compiler.compiler_so[0] = self.mpicc
        self.compiler.linker_so[0] = self.mpicc
        build_ext.build_extensions(self)

extensions = [
        Extension("mpsort.binding", ["mpsort/binding.pyx"],
            include_dirs = ["./", numpy.get_include()])]

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(
    name="mpsort",
    version=find_version("mpsort/version.py"),
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    url="http://github.com/rainwoodman/mpsort",
    description="python binding of MP-sort, a peta scale sorting routine",
    zip_safe = False,
    package_dir = {'mpsort': 'mpsort'},
    install_requires=['cython', 'numpy', 'mpi4py'],
    packages= ['mpsort', 'mpsort.tests'],
    license='BSD-2-Clause',
    cmdclass = {
        "build_ext": build_ext_subclass
    },
    ext_modules = cythonize(extensions)
)
