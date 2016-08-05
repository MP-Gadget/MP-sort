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

def sort(source, orderby, out=None, comm=None):
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

    data1 = numpy.empty(
            len(source), dtype=[
                ('data', (source.dtype, source.shape[1:])),
                ('index', (key.dtype, key.shape[1:]))])

    data1['data'][...] = source
    data1['index'][...] = key

    if out is None:
        out = source
        _sort(data1, orderby='index', comm=comm)
        out[...] = data1['data'][...]
    else:
        data2 = numpy.empty(
            len(out), dtype=[
                ('data', (out.dtype, out.shape[1:])),
                ('index', (key.dtype, key.shape[1:]))])
        _sort(data1, orderby='index', out=data2, comm=comm)
        out[...] = data2['data'][...]

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

    source_size = comm.allreduce(source.size)
    argindex_size = comm.allreduce(argindex.size)

    if source_size != argindex_size:
        raise ValueError("Global size of source and argindex is different")

    if source_size < 1024 * 1024 * 1024:
        inddtype = 'i4'
    else:
        inddtype = 'i8'

    start = sum(comm.allgather(argindex.size)[:comm.rank])
    end = start + argindex.size
    dest = numpy.empty(argindex.size, (source.dtype, source.shape[1:]))

    originind = numpy.arange(start, end, dtype=inddtype)
    originind2 = numpy.empty(source.size, dtype=inddtype)

    sort(originind, orderby=argindex, out=originind2, comm=comm)
    sort(source, orderby=originind2, out=dest, comm=comm)
    return dest


from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
