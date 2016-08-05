from .binding import sort as _sort

import numpy
from numpy.lib.recfunctions import append_fields

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring

def sort(source, orderby=None, out=None, comm=None):
    """
        Sort source array with orderby as the key.
        Store result to out.

        Parameters
        ----------
        source : array, 1d, distributed

        orderby : array, 1d, distributed or string.
            Only integer types are supported.
            must be on the same partition as that of source.
            If orderby is string, it refers to the field in source.

        out : array, 1d distributed
            the total length must be the same as source.
            the itemsize must be the same as source
            if None, the sort is in-place.

        Returns
        -------
            None

    """
    key = orderby
    if isinstance(key, basestring):
        return _sort(source, key, out, comm=comm)

    if key is None:
        D, I = 'DD'
        data1 = numpy.empty(len(source),
                dtype=[('D', (source.dtype, source.shape[1:]))])
        data1['D'][...] = source
    else:
        D, I = 'DI'
        data1 = numpy.empty(len(source),
                dtype=[('D', (source.dtype, source.shape[1:])),
                       ('I', (key.dtype, key.shape[1:]))])

        data1['D'][...] = source
        data1['I'][...] = key

    if out is None:
        out = source
        _sort(data1, orderby=I, comm=comm)
        out[...] = data1[D][...]
    else:
        data2 = numpy.empty(len(out), dtype=data1.dtype)
        _sort(data1, orderby=I, out=data2, comm=comm)
        out[...] = data2[D][...]

def globalrange(array, comm):
    """
        The start and end of local chunk in the global array
    """
    s = comm.allgather(len(array))
    start = sum(s[:comm.rank])
    end = start + s[comm.rank]
    return (start, end)

def globalindices(array, comm):
    """
        Construct a list of indices in the global array.
    """
    start, end = globalrange(array, comm)
    globalsize = comm.bcast(end, root=comm.size - 1)

    if globalsize > 1024 * 1024 * 1024:
        dtype = 'i8'
    else:
        dtype = 'i4'

    return numpy.arange(start, end, dtype=dtype)

def permute(source, argindex, comm):
    """
        Permute a distributed array.

        Constructs source[argindex], distributed as the same partition
        as argindex.

        Parameters
        ----------
        source : array, 1d, distributed
        argindex : array, 1d, distributed

        Returns
        -------
            source[argindex] distributed as the same partition as argindex.

    """

    source_size = comm.allreduce(len(source))
    argindex_size = comm.allreduce(argindex.size)

    if source_size != argindex_size:
        raise ValueError("Global size of source and argindex is different")

    dest = numpy.empty(argindex.size, (source.dtype, source.shape[1:]))

    originind = globalindices(argindex, comm)
    originind2 = numpy.empty(len(source), dtype=originind.dtype)

    sort(originind, orderby=argindex, out=originind2, comm=comm)
    sort(source, orderby=originind2, out=dest, comm=comm)
    return dest

def histogram(array, bins, comm, right=False):
    """
        Histogram of array.

        Parameters
        ----------
        bins : collective, array, 1d

        array : distributed, array, 1d

    """
    originrank = numpy.digitize(array, bins, right)
    recv = numpy.bincount(originrank, minlength=len(bins) + 1)
    return comm.allreduce(recv)

def take(source, argindex, comm):
    """ Take argindex from source.

        source[argindex] distributed as argindex. argindex does not have to
        be a permutation, nor unique.

    """
    nlocal = len(source)
    start, end = globalrange(source, comm)

    # find number of selections made from my rank
    bins = comm.allgather(end)
    h = histogram(argindex, bins, comm)
    # nactive has number of selected objects from this rank.
    nactive = h[comm.rank]

    originind = globalindices(argindex, comm)

    myargindex = numpy.empty(nactive, dtype=argindex.dtype)
    myoriginind = numpy.empty(nactive, dtype=originind.dtype)

    sort(originind, orderby=argindex, out=myoriginind, comm=comm)
    sort(argindex, orderby=argindex, out=myargindex, comm=comm)

    myresult = source[myargindex - start]
    result = numpy.empty(len(argindex), dtype=(source.dtype, source.shape[1:]))

    sort(myresult, orderby=myoriginind, out=result, comm=comm)
    return result

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
