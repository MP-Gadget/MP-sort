from binding import *
from mpi4py import MPI
import numpy
data = numpy.empty(1000, dtype=[('data', 'f4'), ('radix', ('i4', 2))])

data['data'] = numpy.arange(1000)[::-1]
data['radix'][:, 1] = numpy.arange(1000)[::-1]
data['radix'][:, 0] = -MPI.COMM_WORLD.rank

sort(data, orderby='radix')
alldata = MPI.COMM_WORLD.allgather(data)
alldata = numpy.concatenate(alldata)
if MPI.COMM_WORLD.rank == 0:
    print alldata['data']
    print alldata['radix']
    assert numpy.allclose(
            alldata['data'].reshape(-1, MPI.COMM_WORLD.size).mean(axis=-1),
            alldata['data'].reshape(-1, MPI.COMM_WORLD.size)[:, 0])
