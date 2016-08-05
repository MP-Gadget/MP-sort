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


from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
