import mpsort
import numpy
from numpy.testing import assert_array_equal
from itertools import product
import pytest
from mpi4py import MPI

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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sort_i4(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000 - 400)

    local = split(s, comm)
    s = heal(local, comm)

    mpsort.sort(local, orderby=None, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sort_i8(comm):
    s = numpy.int64(numpy.random.random(size=1000) * 1000 - 400)

    local = split(s, comm)
    s = heal(local, comm)

    mpsort.sort(local, orderby=None, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sort_u8(comm):
    s = numpy.uint64(numpy.random.uniform(size=1000, low=-1000000, high=1000000) * 1000 - 400)

    local = split(s, comm)
    s = heal(local, comm)

    mpsort.sort(local, orderby=None, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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
    ['DISABLE_SPARSE_ALLTOALLV'],
    ['REQUIRE_SPARSE_ALLTOALLV'],
    ['REQUIRE_GATHER_SORT'],
    ['DISABLE_GATHER_SORT'],
]

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.parametrize("tuning", TUNINGS)
@pytest.mark.mpi
def test_sort_tunings(comm, tuning):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    g = comm.allgather(local.size)
    mpsort.sort(local, orderby=None, out=None, comm=comm, tuning=tuning)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sort_inplace(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    g = comm.allgather(local.size)
    mpsort.sort(local, local, out=None, comm=comm)

    r = heal(local, comm)
    s.sort()
    assert_array_equal(s, r)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sort_outplace(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    res = numpy.zeros(adjustsize(local.size, comm), dtype=local.dtype)

    mpsort.sort(local, local, out=res, comm=comm)

    s.sort()

    r = heal(res, comm)
    assert_array_equal(s, r)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sort_flatiter(comm):
    s = numpy.int32(numpy.random.random(size=1000) * 1000)

    local = split(s, comm)
    s = heal(local, comm)

    res = numpy.zeros(adjustsize(local.size, comm), dtype=local.dtype)

    mpsort.sort(local.flat, local.flat, out=res.flat, comm=comm)

    s.sort()

    r = heal(res, comm)
    assert_array_equal(s, r)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
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

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_histogram_empty(comm):
    mpsort.histogram([], [1], comm)
    # no error shall be raised

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.parametrize("tuning", TUNINGS)
def test_empty_sort(comm, tuning):
    s = numpy.empty(0, dtype=[
        ('vkey', ('u8', 3)),
        ('vector', ('u4', 3)),
    ])

    mpsort.sort(s, 'vkey', out=s, comm=comm, tuning=tuning)


@pytest.mark.parametrize("comm4", [MPI.COMM_WORLD,])
@pytest.mark.parametrize("sizes, tuning",
   product(product(*([[0, 1, 2]] * 4)), TUNINGS))
@pytest.mark.mpi
def test_few_items(comm4, sizes, tuning):
    comm = comm4
    A = [range(sizes[i]) for i in range(len(sizes)) ]
    s = numpy.empty(len(A[comm.rank]), dtype=[
        ('vkey', ('u8', 3)),
        ('vector', ('u4', 3)),
    ])

    s['vkey'] = numpy.array(A[comm.rank], dtype='u8')[:, None]
    s['vector'] = 1
    S = numpy.concatenate(comm.allgather(s))
    S.sort()
    r = numpy.empty(len(A[comm.rank]), dtype=s.dtype)
    mpsort.sort(s, 'vkey', out=r, comm=comm, tuning=tuning)
    R = numpy.concatenate(comm.allgather(r))
    comm.barrier()
    assert_array_equal(R['vkey'], S['vkey'])
    comm.barrier()


Issue7B64 = b"""
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
BAsAAAByGwEAAIZyHAEAAFJyHQEAAGgEaAhDBBoBAAByHgEAAIZyHwEAAFJyIAEAAGgEaAhDBFcA
AAByIQEAAIZyIgEAAFJyIwEAAGgEaAhDBAcAAAByJAEAAIZyJQEAAFJyJgEAAGgEaAhDBC5mAQBy
JwEAAIZyKAEAAFJyKQEAAGgEaAhDBAAAAAByKgEAAIZyKwEAAFJyLAEAAGgEaAhDBAoAAAByLQEA
AIZyLgEAAFJyLwEAAGgEaAhDBNDstERyMAEAAIZyMQEAAFJyMgEAAGgEaAhDBPAB20RyMwEAAIZy
NAEAAFJyNQEAAGgEaAhDBOn5K0VyNgEAAIZyNwEAAFJyOAEAAGVdcjkBAABdcjoBAAAoaARoCEME
BAAAAHI7AQAAhnI8AQAAUnI9AQAAaARoCEMEAAAAAHI+AQAAhnI/AQAAUnJAAQAAaARoCEMEVQAA
AHJBAQAAhnJCAQAAUnJDAQAAaARoCEMECAAAAHJEAQAAhnJFAQAAUnJGAQAAaARoCEMEDaADAHJH
AQAAhnJIAQAAUnJJAQAAaARoCEMEAAAAAHJKAQAAhnJLAQAAUnJMAQAAaARoCEMEBAAAAHJNAQAA
hnJOAQAAUnJPAQAAaARoCEMEtxBiRXJQAQAAhnJRAQAAUnJSAQAAaARoCEMEmNPeQnJTAQAAhnJU
AQAAUnJVAQAAaARoCEMEvqJvRHJWAQAAhnJXAQAAUnJYAQAAaARoCEMECgAAAHJZAQAAhnJaAQAA
UnJbAQAAaARoCEMEAAAAAHJcAQAAhnJdAQAAUnJeAQAAaARoCEMETwAAAHJfAQAAhnJgAQAAUnJh
AQAAaARoCEMECQAAAHJiAQAAhnJjAQAAUnJkAQAAaARoCEMEvLsAAHJlAQAAhnJmAQAAUnJnAQAA
aARoCEMEAAAAAHJoAQAAhnJpAQAAUnJqAQAAaARoCEMECgAAAHJrAQAAhnJsAQAAUnJtAQAAaARo
CEMEmWRxRHJuAQAAhnJvAQAAUnJwAQAAaARoCEMEfAY/RXJxAQAAhnJyAQAAUnJzAQAAaARoCEME
YCtxRXJ0AQAAhnJ1AQAAUnJ2AQAAaARoCEMECQAAAHJ3AQAAhnJ4AQAAUnJ5AQAAaARoCEMErwAA
AHJ6AQAAhnJ7AQAAUnJ8AQAAaARoCEMETwAAAHJ9AQAAhnJ+AQAAUnJ/AQAAaARoCEMECQAAAHKA
AQAAhnKBAQAAUnKCAQAAaARoCEMEvLsAAHKDAQAAhnKEAQAAUnKFAQAAaARoCEMEAAAAAHKGAQAA
hnKHAQAAUnKIAQAAaARoCEMECgAAAHKJAQAAhnKKAQAAUnKLAQAAaARoCEMEmWRxRHKMAQAAhnKN
AQAAUnKOAQAAaARoCEMEfAY/RXKPAQAAhnKQAQAAUnKRAQAAaARoCEMEYCtxRXKSAQAAhnKTAQAA
UnKUAQAAaARoCEMEAwAAAHKVAQAAhnKWAQAAUnKXAQAAaARoCEMEAAAAAHKYAQAAhnKZAQAAUnKa
AQAAaARoCEMETAAAAHKbAQAAhnKcAQAAUnKdAQAAaARoCEMECgAAAHKeAQAAhnKfAQAAUnKgAQAA
aARoCEMEUCsCAHKhAQAAhnKiAQAAUnKjAQAAaARoCEMEAAAAAHKkAQAAhnKlAQAAUnKmAQAAaARo
CEMEAwAAAHKnAQAAhnKoAQAAUnKpAQAAaARoCEMEcMwTRXKqAQAAhnKrAQAAUnKsAQAAaARoCEME
HfgvRXKtAQAAhnKuAQAAUnKvAQAAaARoCEMEgOOARHKwAQAAhnKxAQAAUnKyAQAAZV1yswEAAF1y
tAEAAF1ytQEAAChoBGgIQwQKAAAAcrYBAACGcrcBAABScrgBAABoBGgIQwQAAAAAcrkBAACGcroB
AABScrsBAABoBGgIQwRGAAAAcrwBAACGcr0BAABScr4BAABoBGgIQwQLAAAAcr8BAACGcsABAABS
csEBAABoBGgIQwRtNgEAcsIBAACGcsMBAABScsQBAABoBGgIQwQAAAAAcsUBAACGcsYBAABScscB
AABoBGgIQwQKAAAAcsgBAACGcskBAABScsoBAABoBGgIQwSusrFEcssBAACGcswBAABScs0BAABo
BGgIQwQGFNtEcs4BAACGcs8BAABSctABAABoBGgIQwSZnSxFctEBAACGctIBAABSctMBAABoBGgI
QwQLAAAActQBAACGctUBAABSctYBAABoBGgIQwQAAAAActcBAACGctgBAABSctkBAABoBGgIQwRF
AAAActoBAACGctsBAABSctwBAABoBGgIQwQMAAAAct0BAACGct4BAABSct8BAABoBGgIQwRtdgEA
cuABAACGcuEBAABScuIBAABoBGgIQwQAAAAAcuMBAACGcuQBAABScuUBAABoBGgIQwQLAAAAcuYB
AACGcucBAABScugBAABoBGgIQwQhn79EcukBAACGcuoBAABScusBAABoBGgIQwRNoNtEcuwBAACG
cu0BAABScu4BAABoBGgIQwS6vyhFcu8BAACGcvABAABScvEBAABoBGgIQwQJAAAAcvIBAACGcvMB
AABScvQBAABoBGgIQwQAAAAAcvUBAACGcvYBAABScvcBAABoBGgIQwQ+AAAAcvgBAACGcvkBAABS
cvoBAABoBGgIQwQNAAAAcvsBAACGcvwBAABScv0BAABoBGgIQwS6bwEAcv4BAACGcv8BAABScgAC
AABoBGgIQwQAAAAAcgECAACGcgICAABScgMCAABoBGgIQwQJAAAAcgQCAACGcgUCAABScgYCAABo
BGgIQwRzXMREcgcCAACGcggCAABScgkCAABoBGgIQwTnKXVFcgoCAACGcgsCAABScgwCAABoBGgI
QwT2emhFcg0CAACGcg4CAABScg8CAABoBGgIQwQDAAAAchACAACGchECAABSchICAABoBGgIQwQA
AAAAchMCAACGchQCAABSchUCAABoBGgIQwQ8AAAAchYCAACGchcCAABSchgCAABoBGgIQwQOAAAA
chkCAACGchoCAABSchsCAABoBGgIQwSWWgIAchwCAACGch0CAABSch4CAABoBGgIQwQAAAAAch8C
AACGciACAABSciECAABoBGgIQwQDAAAAciICAACGciMCAABSciQCAABoBGgIQwSLtB1FciUCAACG
ciYCAABScicCAABoBGgIQwQp4CdFcigCAACGcikCAABScioCAABoBGgIQwTJa6JEcisCAACGciwC
AABSci0CAABoBGgIQwQKAAAAci4CAACGci8CAABScjACAABoBGgIQwQAAAAAcjECAACGcjICAABS
cjMCAABoBGgIQwQ6AAAAcjQCAACGcjUCAABScjYCAABoBGgIQwQPAAAAcjcCAACGcjgCAABScjkC
AABoBGgIQwQndgEAcjoCAACGcjsCAABScjwCAABoBGgIQwQAAAAAcj0CAACGcj4CAABScj8CAABo
BGgIQwQKAAAAckACAACGckECAABSckICAABoBGgIQwQ2QcdEckMCAACGckQCAABSckUCAABoBGgI
QwRX09ZEckYCAACGckcCAABSckgCAABoBGgIQwSWKRtFckkCAACGckoCAABScksCAABoBGgIQwQJ
AAAAckwCAACGck0CAABSck4CAABoBGgIQwQAAAAAck8CAACGclACAABSclECAABoBGgIQwQ4AAAA
clICAACGclMCAABSclQCAABoBGgIQwQQAAAAclUCAACGclYCAABSclcCAABoBGgIQwR4HgEAclgC
AACGclkCAABScloCAABoBGgIQwQAAAAAclsCAACGclwCAABScl0CAABoBGgIQwQJAAAAcl4CAACG
cl8CAABScmACAABoBGgIQwRi46NEcmECAACGcmICAABScmMCAABoBGgIQwS4m2hFcmQCAACGcmUC
AABScmYCAABoBGgIQwR09WJFcmcCAACGcmgCAABScmkCAABlXXJqAgAAKGgEaAhDBAoAAAByawIA
AIZybAIAAFJybQIAAGgEaAhDBAAAAABybgIAAIZybwIAAFJycAIAAGgEaAhDBDYAAABycQIAAIZy
cgIAAFJycwIAAGgEaAhDBBEAAABydAIAAIZydQIAAFJydgIAAGgEaAhDBOelAQBydwIAAIZyeAIA
AFJyeQIAAGgEaAhDBAAAAAByegIAAIZyewIAAFJyfAIAAGgEaAhDBAoAAAByfQIAAIZyfgIAAFJy
fwIAAGgEaAhDBFsb1kRygAIAAIZygQIAAFJyggIAAGgEaAhDBKTi1kRygwIAAIZyhAIAAFJyhQIA
AGgEaAhDBALxF0VyhgIAAIZyhwIAAFJyiAIAAGgEaAhDBAoAAAByiQIAAIZyigIAAFJyiwIAAGgE
aAhDBAAAAAByjAIAAIZyjQIAAFJyjgIAAGgEaAhDBDMAAAByjwIAAIZykAIAAFJykQIAAGgEaAhD
BBIAAABykgIAAIZykwIAAFJylAIAAGgEaAhDBOSlAQBylQIAAIZylgIAAFJylwIAAGgEaAhDBAAA
AABymAIAAIZymQIAAFJymgIAAGgEaAhDBAoAAABymwIAAIZynAIAAFJynQIAAGgEaAhDBLhS3kRy
ngIAAIZynwIAAFJyoAIAAGgEaAhDBJXq0ERyoQIAAIZyogIAAFJyowIAAGgEaAhDBKW5DkVypAIA
AIZypQIAAFJypgIAAGgEaAhDBAoAAABypwIAAIZyqAIAAFJyqQIAAGgEaAhDBAAAAAByqgIAAIZy
qwIAAFJyrAIAAGgEaAhDBDIAAAByrQIAAIZyrgIAAFJyrwIAAGgEaAhDBBMAAABysAIAAIZysQIA
AFJysgIAAGgEaAhDBGa3AAByswIAAIZytAIAAFJytQIAAGgEaAhDBAAAAABytgIAAIZytwIAAFJy
uAIAAGgEaAhDBAoAAAByuQIAAIZyugIAAFJyuwIAAGgEaAhDBJp7cERyvAIAAIZyvQIAAFJyvgIA
AGgEaAhDBIBk8URyvwIAAIZywAIAAFJywQIAAGgEaAhDBCzMGEVywgIAAIZywwIAAFJyxAIAAGgE
aAhDBAMAAAByxQIAAIZyxgIAAFJyxwIAAGgEaAhDBAAAAAByyAIAAIZyyQIAAFJyygIAAGgEaAhD
BDAAAAByywIAAIZyzAIAAFJyzQIAAGgEaAhDBBQAAAByzgIAAIZyzwIAAFJy0AIAAGgEaAhDBJlq
AgBy0QIAAIZy0gIAAFJy0wIAAGgEaAhDBAAAAABy1AIAAIZy1QIAAFJy1gIAAGgEaAhDBAMAAABy
1wIAAIZy2AIAAFJy2QIAAGgEaAhDBPpWIkVy2gIAAIZy2wIAAFJy3AIAAGgEaAhDBDQqJUVy3QIA
AIZy3gIAAFJy3wIAAGgEaAhDBHB+t0Ry4AIAAIZy4QIAAFJy4gIAAGgEaAhDBAMAAABy4wIAAIZy
5AIAAFJy5QIAAGgEaAhDBAAAAABy5gIAAIZy5wIAAFJy6AIAAGgEaAhDBC4AAABy6QIAAIZy6gIA
AFJy6wIAAGgEaAhDBBUAAABy7AIAAIZy7QIAAFJy7gIAAGgEaAhDBFXIAgBy7wIAAIZy8AIAAFJy
8QIAAGgEaAhDBAAAAABy8gIAAIZy8wIAAFJy9AIAAGgEaAhDBAMAAABy9QIAAIZy9gIAAFJy9wIA
AGgEaAhDBE4mM0Vy+AIAAIZy+QIAAFJy+gIAAGgEaAhDBJomBEVy+wIAAIZy/AIAAFJy/QIAAGgE
aAhDBFmHpkRy/gIAAIZy/wIAAFJyAAMAAGgEaAhDBAoAAAByAQMAAIZyAgMAAFJyAwMAAGgEaAhD
BAAAAAByBAMAAIZyBQMAAFJyBgMAAGgEaAhDBCwAAAByBwMAAIZyCAMAAFJyCQMAAGgEaAhDBBYA
AAByCgMAAIZyCwMAAFJyDAMAAGgEaAhDBKTVAQByDQMAAIZyDgMAAFJyDwMAAGgEaAhDBAAAAABy
EAMAAIZyEQMAAFJyEgMAAGgEaAhDBAoAAAByEwMAAIZyFAMAAFJyFQMAAGgEaAhDBLSH8ERyFgMA
AIZyFwMAAFJyGAMAAGgEaAhDBC5l0URyGQMAAIZyGgMAAFJyGwMAAGgEaAhDBGq5DEVyHAMAAIZy
HQMAAFJyHgMAAGgEaAhDBAMAAAByHwMAAIZyIAMAAFJyIQMAAGgEaAhDBAAAAAByIgMAAIZyIwMA
AFJyJAMAAGgEaAhDBCsAAAByJQMAAIZyJgMAAFJyJwMAAGgEaAhDBBcAAAByKAMAAIZyKQMAAFJy
KgMAAGgEaAhDBFXZAgByKwMAAIZyLAMAAFJyLQMAAGgEaAhDBAAAAAByLgMAAIZyLwMAAFJyMAMA
AGgEaAhDBAMAAAByMQMAAIZyMgMAAFJyMwMAAGgEaAhDBP9gM0VyNAMAAIZyNQMAAFJyNgMAAGgE
aAhDBOBAGUVyNwMAAIZyOAMAAFJyOQMAAGgEaAhDBJ+ZqURyOgMAAIZyOwMAAFJyPAMAAGgEaAhD
BAsAAAByPQMAAIZyPgMAAFJyPwMAAGgEaAhDBAAAAAByQAMAAIZyQQMAAFJyQgMAAGgEaAhDBCoA
AAByQwMAAIZyRAMAAFJyRQMAAGgEaAhDBBgAAAByRgMAAIZyRwMAAFJySAMAAGgEaAhDBDnQAQBy
SQMAAIZySgMAAFJySwMAAGgEaAhDBAAAAAByTAMAAIZyTQMAAFJyTgMAAGgEaAhDBAsAAAByTwMA
AIZyUAMAAFJyUQMAAGgEaAhDBMq65ERyUgMAAIZyUwMAAFJyVAMAAGgEaAhDBNMJWUJyVQMAAIZy
VgMAAFJyVwMAAGgEaAhDBOe9ZEVyWAMAAIZyWQMAAFJyWgMAAGgEaAhDBAoAAAByWwMAAIZyXAMA
AFJyXQMAAGgEaAhDBAAAAAByXgMAAIZyXwMAAFJyYAMAAGgEaAhDBCoAAAByYQMAAIZyYgMAAFJy
YwMAAGgEaAhDBBkAAAByZAMAAIZyZQMAAFJyZgMAAGgEaAhDBCb2AQByZwMAAIZyaAMAAFJyaQMA
AGgEaAhDBAAAAAByagMAAIZyawMAAFJybAMAAGgEaAhDBAoAAABybQMAAIZybgMAAFJybwMAAGgE
aAhDBNQW9URycAMAAIZycQMAAFJycgMAAGgEaAhDBLi01ERycwMAAIZydAMAAFJydQMAAGgEaAhD
BEf/DkVydgMAAIZydwMAAFJyeAMAAGgEaAhDBAkAAAByeQMAAIZyegMAAFJyewMAAGgEaAhDBAAA
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
hnJHBAAAUnJIBAAAaARoCEMEIm4VRXJJBAAAhnJKBAAAUnJLBAAAaARoCEMECgAAAHJMBAAAhnJN
BAAAUnJOBAAAaARoCEMEAAAAAHJPBAAAhnJQBAAAUnJRBAAAaARoCEMEIAAAAHJSBAAAhnJTBAAA
UnJUBAAAaARoCEMEIAAAAHJVBAAAhnJWBAAAUnJXBAAAaARoCEMEbQcBAHJYBAAAhnJZBAAAUnJa
BAAAaARoCEMEAAAAAHJbBAAAhnJcBAAAUnJdBAAAaARoCEMECgAAAHJeBAAAhnJfBAAAUnJgBAAA
aARoCEMEOfGiRHJhBAAAhnJiBAAAUnJjBAAAaARoCEMEqoPmRHJkBAAAhnJlBAAAUnJmBAAAaARo
CEMECxwpRXJnBAAAhnJoBAAAUnJpBAAAaARoCEMECwAAAHJqBAAAhnJrBAAAUnJsBAAAaARoCEME
AAAAAHJtBAAAhnJuBAAAUnJvBAAAaARoCEMEIAAAAHJwBAAAhnJxBAAAUnJyBAAAaARoCEMEIQAA
AHJzBAAAhnJ0BAAAUnJ1BAAAaARoCEMEPPABAHJ2BAAAhnJ3BAAAUnJ4BAAAaARoCEMEAAAAAHJ5
BAAAhnJ6BAAAUnJ7BAAAaARoCEMECwAAAHJ8BAAAhnJ9BAAAUnJ+BAAAaARoCEMEsF3vRHJ/BAAA
hnKABAAAUnKBBAAAaARoCEMEqyfRQnKCBAAAhnKDBAAAUnKEBAAAaARoCEMErNtmRXKFBAAAhnKG
BAAAUnKHBAAAaARoCEMEAwAAAHKIBAAAhnKJBAAAUnKKBAAAaARoCEMEAAAAAHKLBAAAhnKMBAAA
UnKNBAAAaARoCEMEIAAAAHKOBAAAhnKPBAAAUnKQBAAAaARoCEMEIgAAAHKRBAAAhnKSBAAAUnKT
BAAAaARoCEMEVaoCAHKUBAAAhnKVBAAAUnKWBAAAaARoCEMEAAAAAHKXBAAAhnKYBAAAUnKZBAAA
aARoCEMEAwAAAHKaBAAAhnKbBAAAUnKcBAAAaARoCEMEvgIoRXKdBAAAhnKeBAAAUnKfBAAAaARo
CEMEkAYnRXKgBAAAhnKhBAAAUnKiBAAAaARoCEMEwrirRHKjBAAAhnKkBAAAUnKlBAAAZWUu
"""
@pytest.mark.parametrize("tuning", TUNINGS)
@pytest.mark.parametrize("comm12", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_issue7(comm12, tuning):
    # This roughly mimics the behavior of issue7, with data size of 40 bytes;
    # and radix size of 16 bytes, with 8 bytes from offset 0, and 4 bytes from offset 16.
    comm = comm12
    import base64
    import pickle
    A = [numpy.array(a, dtype='u4').reshape(-1, 10) for a in pickle.loads(base64.decodebytes(Issue7B64))]

    s = numpy.zeros(len(A[comm.rank]), dtype=[('radix', ('u8', 2)), ('ext', ('u8', 3))])

    s['radix'][:, 0] = (A[comm.rank][:, 5]) + (A[comm.rank][:, 4] << 32)
    s['radix'][:, 1] = A[comm.rank][:, 0]

    S = numpy.concatenate(comm.allgather(s))
    ind = numpy.lexsort(S['radix'].T)
    S = S[ind]
    r = numpy.empty(len(A[comm.rank]), dtype=s.dtype)
    mpsort.sort(s, orderby='radix', out=r, comm=comm, tuning=tuning)
    R = numpy.concatenate(comm.allgather(r))

    comm.barrier()
    assert_array_equal(R.flatten(), S.flatten())
    comm.barrier()

