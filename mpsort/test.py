from binding import *
from mpi4py import MPI
import numpy

NperRank = 4
data = numpy.empty(NperRank, dtype=[('data', 'f4'), ('radix', ('i4', 2))])

Nnew = NperRank
if MPI.COMM_WORLD.rank == 1:
    Nnew = NperRank / 2
    pass
elif MPI.COMM_WORLD.rank == 0:
    Nnew = NperRank + NperRank / 2
    pass

out = numpy.empty(Nnew, dtype=[('data', 'f4'), ('radix', ('i4', 2))])

data['data'] = numpy.arange(NperRank)[::-1]
data['radix'][:, 0] = numpy.arange(NperRank)[::-1]
data['radix'][:, 1] = -MPI.COMM_WORLD.rank

sort(data, orderby='radix', out=out)
alldata = MPI.COMM_WORLD.allgather(out)
alldata = numpy.concatenate(alldata)
if MPI.COMM_WORLD.rank == 0:
    print alldata['data']
    print alldata['radix']
    print alldata['data'].reshape(-1, NperRank).sum(axis=-1)
    assert numpy.allclose(
        alldata['data'].reshape(-1, NperRank).sum(axis=-1), 
        NperRank * (NperRank - 1) / 2
        )

sort(data, orderby='radix')
alldata = MPI.COMM_WORLD.allgather(out)
alldata = numpy.concatenate(alldata)
if MPI.COMM_WORLD.rank == 0:
    print alldata['data']
    print alldata['radix']
    print alldata['data'].reshape(-1, NperRank).sum(axis=-1)
    assert numpy.allclose(
        alldata['data'].reshape(-1, NperRank).sum(axis=-1), 
        NperRank * (NperRank - 1) / 2
        )
