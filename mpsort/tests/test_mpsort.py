import mpsort
from mpi4py import MPI
import numpy
from numpy.testing import assert_array_equal

def MPIWorld(NTask, required=[]):
    if not isinstance(NTask, (tuple, list)):
        NTask = (NTask,)

    if not isinstance(required, (tuple, list)):
        required = (required,)

    maxsize = max(required)
    if MPI.COMM_WORLD.size < maxsize:
        raise ValueError("Test Failed because the world is too small. Increase to mpirun -n %d" % maxsize)

    def dec(func):
        def wrapped(*args, **kwargs):
            for size in sorted(set(list(required) + list(NTask))):
                if MPI.COMM_WORLD.size < size: continue
                color = MPI.COMM_WORLD.rank >= size
                comm = MPI.COMM_WORLD.Split(color)

                kwargs['comm'] = comm
                if color == 0:
                    func(*args, **kwargs)

        wrapped.__name__ = func.__name__
        return wrapped
    return dec

@MPIWorld(NTask=(1, 2, 3, 4), required=1)
def test_sort_inplace(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)
    s = comm.bcast(s)
    local = comm.scatter(numpy.array_split(s, comm.size))

    mpsort.sort(local, local, out=local, comm=comm)

    r = numpy.concatenate(comm.allgather(local))
    s.sort()
    assert_array_equal(s, r)

@MPIWorld(NTask=(1, 2, 3, 4), required=1)
def test_sort_outplace(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)
    s = comm.bcast(s)

    local = comm.scatter(numpy.array_split(s.copy(), comm.size))
    ressize = local.size + 1 - 2 * ((comm.rank) % 2)
    if comm.size % 2 == 1:
        if comm.rank == 0:
            ressize = local.size

    res = numpy.zeros(ressize, dtype=local.dtype)

    mpsort.sort(local, local, out=res, comm=comm)

    r = comm.allgather(res)
    s.sort()
    r = numpy.concatenate(r)
    assert_array_equal(s, r)

@MPIWorld(NTask=(1, 2, 3, 4), required=1)
def test_sort_struct(comm):
    s = numpy.empty(1000, dtype=[
        ('value', 'i8'),
        ('key', 'i8')])

    s['value'] = numpy.int32(numpy.random.random(size=1000) * 1000)
    s['key'] = s['value']

    s = comm.bcast(s)

    local = comm.scatter(numpy.array_split(s, comm.size))
    res = numpy.zeros_like(local)

    mpsort.sort(local, 'key', out=res, comm=comm)

    r = numpy.concatenate(comm.allgather(res))
    s.sort(order='key')
    assert_array_equal(s['value'], r['value'])

@MPIWorld(NTask=(1, 2, 3, 4), required=1)
def test_sort_struct_vector(comm):
    s = numpy.empty(10, dtype=[
        ('value', 'i8'),
        ('key', 'i8'),
        ('vkey', ('i8', 2))])

    s['value'] = numpy.int32(numpy.random.random(size=len(s)) * 1000)

    # use a scalar key to trick numpy
    # numpy sorts as byte streams for vector fields.
    s['key'][:][...] = s['value']
    s['vkey'][:, 0][...] = s['value']
    s['vkey'][:, 1][...] = s['value']

    s = comm.bcast(s)

    local = comm.scatter(numpy.array_split(s, comm.size))
    res = numpy.zeros_like(local)

    mpsort.sort(local, 'vkey', out=res, comm=comm)

    r = numpy.concatenate(comm.allgather(res))
    s.sort(order='key')
    assert_array_equal(s['value'], r['value'])

@MPIWorld(NTask=(1, 2, 3, 4), required=1)
def test_sort_vector(comm):
    s = numpy.empty(10, dtype=[('value', 'i8')])

    s['value'] = numpy.int32(numpy.random.random(size=len(s)) * 1000)
    s = comm.bcast(s)

    local = comm.scatter(numpy.array_split(s, comm.size))

    k = numpy.empty(len(local), ('i8', 2))
    k[:, 0][...] = local['value']
    k[:, 1][...] = local['value']

    res = numpy.zeros_like(local)

    mpsort.sort(local, k, out=res, comm=comm)

    r = numpy.concatenate(comm.allgather(res))
    s.sort(order='value')
    assert_array_equal(s['value'], r['value'])
