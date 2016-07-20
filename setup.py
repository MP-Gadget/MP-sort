from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

import mpi4py
import os
if 'MPICC' in os.environ:
    compiler = os.environ['MPICC']
else:
    try:
        compiler = str(mpi4py.get_config()['mpicc'])
    except:
        pass
    compiler = "mpicc"

os.environ['CC'] = compiler

if 'LDSHARED' not in os.environ:
    os.environ['LDSHARED'] = compiler + ' -shared'

extensions = [
        Extension("mpsort.binding", ["mpsort/binding.pyx"],
            include_dirs = ["./", numpy.get_include()])]

setup(
    name="mpsort", version="0.1.3",
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    url="http://github.com/rainwoodman/mpsort",
    description="python binding of MP-sort, a peta scale sorting routine",
    zip_safe = False,
    package_dir = {'mpsort': 'mpsort'},
    install_requires=['cython', 'numpy'],
    packages= ['mpsort'],
    requires=['numpy'],
    ext_modules = cythonize(extensions)
)
