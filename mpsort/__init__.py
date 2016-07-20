from .binding import sort as _sort

import numpy
from numpy.lib.recfunctions import append_fields

def sort(source, orderby, out=None, comm=None):
    key = orderby
    if isinstance(key, basestring):
        return _sort(source, key, out, comm=comm)

    data1 = numpy.empty(
            len(source), dtype=[
                ('data', (source.dtype, source.shape[1:])),
                ('index', (key.dtype, key.shape[1:]))])

    data1['data'][...] = source
    data1['index'][...] = key

    _sort(data1, orderby='index', comm=comm)

    if out is None:
        out = source

    out[...] = data1['data'][...]
