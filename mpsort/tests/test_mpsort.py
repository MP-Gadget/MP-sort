import mpsort
from runtests.mpi import MPITest
import numpy
from numpy.testing import assert_array_equal
import pytest

def split(array, comm, localsize=None):
    array = comm.bcast(array)
    if localsize is None:
        sp = numpy.array_split(array, comm.size)
        return comm.scatter(sp)
    else:
        g = comm.allgather(localsize)
        return comm.scatter(numpy.array_split(array, numpy.cumsum(g)[:-1]))

def heal(array, comm):
    a = comm.allgather(array)
    a = numpy.concatenate(a, axis=0)
    return a

def adjustsize(size, comm):
    ressize = size + 1 - 2 * ((comm.rank) % 2)
    if comm.size % 2 == 1:
        if comm.rank == 0:
            ressize = size
    return ressize

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_i4(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000 - 400)

    local = split(s, comm)
    s = heal(local, comm)

    mpsort.sort(local, orderby=None, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_i8(comm):
    s = numpy.int64(numpy.random.random(size=1000) * 1000 - 400)

    local = split(s, comm)
    s = heal(local, comm)

    mpsort.sort(local, orderby=None, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_u8(comm):
    s = numpy.uint64(numpy.random.random(size=1000) * 1000 - 400)

    local = split(s, comm)
    s = heal(local, comm)

    mpsort.sort(local, orderby=None, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_u4(comm):
    s = numpy.uint32(numpy.random.random(size=1000) * 1000 - 400)

    local = split(s, comm)
    s = heal(local, comm)

    mpsort.sort(local, orderby=None, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

TUNINGS = [
    [],
    ['ENABLE_SPARSE_ALLTOALLV'],
    ['REQUIRE_SPARSE_ALLTOALLV'],
    ['REQUIRE_GATHER_SORT'],
    ['DISABLE_GATHER_SORT'],
    ['DISABLE_IALLREDUCE'],
]

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_tunings(comm):
    for tuning in TUNINGS:
        s = numpy.int32(numpy.random.random(size=1000) * 1000)

        local = split(s, comm)
        s = heal(local, comm)

        g = comm.allgather(local.size)
        mpsort.sort(local, orderby=None, out=None, comm=comm, tuning=tuning)

        r = heal(local, comm)
        s.sort()
        assert_array_equal(s, r)


@MPITest(commsize=(1, 2, 3, 4))
def test_sort_inplace(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    g = comm.allgather(local.size)
    mpsort.sort(local, local, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@MPITest(commsize=(4))
def test_sort_mismatched_zeros(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm, [0, 400, 0, 600][comm.rank])
    s = heal(local, comm)

    res = split(s, comm, [200, 200, 0, 600][comm.rank])
    res[:] = numpy.int32(numpy.random.random(size=res.size) * 1000)
    mpsort.sort(local, local, out=res, comm=comm, tuning=['REQUIRE_GATHER_SORT'])

    s.sort()

    r = heal(res, comm)
    assert_array_equal(s, r)

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_outplace(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    res = numpy.zeros(adjustsize(local.size, comm), dtype=local.dtype)

    mpsort.sort(local, local, out=res, comm=comm)

    s.sort()

    r = heal(res, comm)
    assert_array_equal(s, r)

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_flatiter(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    res = numpy.zeros(adjustsize(local.size, comm), dtype=local.dtype)

    mpsort.sort(local.flat, local.flat, out=res.flat, comm=comm)

    s.sort()

    r = heal(res, comm)
    assert_array_equal(s, r)

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_struct(comm):
    s = numpy.empty(10, dtype=[
        ('value', 'i8'),
        ('key', 'i8')])

    numpy.random.seed(1234)

    s['value'] = numpy.int32(numpy.random.random(size=10) * 1000-400)
    s['key'] = s['value']

    backup = s.copy()
    local = split(s, comm)
    s = heal(local, comm)

    res = numpy.zeros_like(local)

    mpsort.sort(local, 'key', out=res, comm=comm)

    r = heal(res, comm)

    backup.sort(order='key')
    assert_array_equal(backup['value'], r['value'])

@MPITest(commsize=(1, 2, 3, 4))
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

    local = split(s, comm)
    s = heal(local, comm)

    res = numpy.zeros_like(local)

    mpsort.sort(local, 'vkey', out=res, comm=comm)

    r = heal(res, comm)
    s.sort(order='key')
    assert_array_equal(s['value'], r['value'])

@MPITest(commsize=(1, 2, 3, 4))
def test_sort_vector(comm):
    s = numpy.empty(10, dtype=[('value', 'i8')])

    s['value'] = numpy.int32(numpy.random.random(size=len(s)) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    k = numpy.empty(len(local), ('i8', 2))
    k[:, 0][...] = local['value']
    k[:, 1][...] = local['value']

    res = numpy.zeros_like(local)

    mpsort.sort(local, k, out=res, comm=comm)

    s.sort(order='value')

    r = heal(res, comm)

    assert_array_equal(s['value'], r['value'])


@MPITest(commsize=(1, 2, 3, 4))
def test_permute(comm):
    s = numpy.arange(10)
    local = split(s, comm)
    i = numpy.arange(9, -1, -1)
    ind = split(i, comm, adjustsize(local.size, comm))

    res = mpsort.permute(local, ind, comm)
    r = heal(res, comm)
    s = s[i]
    assert res.size == ind.size
    assert_array_equal(r, s)

@MPITest(commsize=(1, 2, 3, 4))
def test_permute_out(comm):
    s = numpy.arange(10)
    local = split(s, comm)
    i = numpy.arange(9, -1, -1)
    ind = split(i, comm, adjustsize(local.size, comm))

    res = numpy.empty(local.size, local.dtype)
    mpsort.permute(local, ind, comm, out=res)
    r = heal(res, comm)
    s = s[i]
    assert_array_equal(r, s)

@MPITest(commsize=(1, 2, 3, 4))
def test_take(comm):
    s = numpy.arange(10)
    local = split(s, comm)
    i = numpy.arange(9, -1, -1)
    ind = split(i, comm, adjustsize(local.size, comm))

    res = mpsort.take(local, ind, comm)
    r = heal(res, comm)
    s = s[i]
    assert res.size == ind.size
    assert_array_equal(r, s)

@MPITest(commsize=(1, 2, 3, 4))
def test_take_out(comm):
    s = numpy.arange(10)
    local = split(s, comm)
    i = numpy.arange(9, -1, -1)
    ind = split(i, comm, adjustsize(local.size, comm))

    res = numpy.empty(local.size, local.dtype)
    mpsort.take(local, ind, comm, out=res)
    r = heal(res, comm)
    s = s[i]
    assert_array_equal(r, s)

def test_version():
    import mpsort
    assert hasattr(mpsort, "__version__")

@MPITest(commsize=(1, 2, 3, 4))
def test_histogram_empty(comm):
    mpsort.histogram([], [1], comm)
    # no error shall be raised

