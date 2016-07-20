import mpsort
from mpi4py import MPI
import numpy
from numpy.testing import assert_array_equal

comm = MPI.COMM_WORLD

def test_sort_inplace():
    s = numpy.int32(numpy.random.random(size=1000) * 1000)
    s = comm.bcast(s)
    local = comm.scatter(numpy.array_split(s, comm.size))

    mpsort.sort(local, local, out=local, comm=comm)

    r = numpy.concatenate(comm.allgather(local))
    s.sort()
    assert_array_equal(s, r)

def test_sort_outplace():
    s = numpy.int32(numpy.random.random(size=1000) * 1000)
    s = comm.bcast(s)

    local = comm.scatter(numpy.array_split(s.copy(), comm.size))
    res = numpy.zeros_like(local)

    mpsort.sort(local, local, out=res, comm=comm)

    r = comm.allgather(res)
    s.sort()
    r = numpy.concatenate(r)
    assert_array_equal(s, r)

def test_sort_struct():
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

def test_sort_struct_vector():
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

def test_sort_vector():
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
