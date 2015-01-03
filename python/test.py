from binding import *
from mpi4py import MPI
import numpy
data = numpy.empty(1000, dtype=[('data', 'f4'), ('radix', 'u8')])

data['data'] = numpy.arange(1000)[::-1]
data['radix'] = numpy.arange(1000)[::-1]

sort(data, data['radix'])
alldata = MPI.COMM_WORLD.allgather(data)
alldata = numpy.concatenate(alldata)
if MPI.COMM_WORLD.rank == 0:
    assert numpy.allclose(
            alldata['data'].reshape(-1, MPI.COMM_WORLD.size).mean(axis=-1),
            alldata['data'].reshape(-1, MPI.COMM_WORLD.size)[:, 0])
    assert numpy.allclose(
            alldata['radix'].reshape(-1, MPI.COMM_WORLD.size).mean(axis=-1),
            alldata['radix'].reshape(-1, MPI.COMM_WORLD.size)[:, 0])
