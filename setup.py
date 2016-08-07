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
    install_requires=['cython', 'numpy'],
    packages= ['mpsort', 'mpsort.tests'],
    requires=['numpy'],
    ext_modules = cythonize(extensions)
)
