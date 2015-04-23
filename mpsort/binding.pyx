#cython: embedsignature=True
cimport numpy
cimport libmpi as MPI
from libc.stddef cimport ptrdiff_t
from libc.stdint cimport uint64_t, int64_t
from libc.string cimport memcpy
import numpy
from mpi4py import MPI as pyMPI

cdef extern from "radixsort.c":
    pass
cdef extern from "radixsort-mpi.c":
    void radix_sort_mpi(void * base, size_t nmemb, size_t size,
            void (*radix)(void * ptr, void * radix, void * arg), 
            size_t rsize, 
            void * arg, MPI.MPI_Comm comm)

cdef class MyClosure:
    cdef void * base
    cdef int elsize
    cdef numpy.uint64_t [:, :] radixarray_u8
    cdef numpy.int64_t [:, :] radixarray_i8
    cdef numpy.uint32_t [:, :] radixarray_u4
    cdef numpy.int32_t [:, :] radixarray_i4
    cdef void (*myradix)(void * ptr, void * radix, void * arg)
    cdef int radix_length

    def __init__(self, numpy.ndarray dataarray, radixkey):
        self.base = dataarray.data
        self.elsize = dataarray.strides[0]
        if radixkey is not None:
            radixarray = dataarray[radixkey]
        else:
            radixarray = dataarray

        if len(radixarray.shape) == 1:
            radixarray = radixarray.reshape(-1, 1)
        elif len(radixarray.shape) == 2:
            radixarray = radixarray
        else:
            raise ValueError("data[%s] is not 1d nor 2d %s" % (radixkey,
                str(dataarray[radixkey].shape)))
        self.radix_length = radixarray.shape[1]

        if radixarray.dtype == numpy.dtype('u8'):
            self.myradix = myradix_u8
            self.radixarray_u8 = radixarray

        elif radixarray.dtype == numpy.dtype('i8'):
            self.myradix = myradix_i8
            self.radixarray_i8 = radixarray
        elif radixarray.dtype == numpy.dtype('u4'):
            self.myradix = myradix_u4
            self.radixarray_u4 = radixarray

        elif radixarray.dtype == numpy.dtype('i4'):
            self.myradix = myradix_i4
            self.radixarray_i4 = radixarray
        else:
            raise TypeError("data[%s] is not u8 or i8" % (radixkey))

cdef void myradix_u8(void * ptr, void * radix, void * arg):
    cdef MyClosure clo = <MyClosure> arg
    cdef numpy.intp_t ind
    cdef int i
    ind = (<char*>ptr - <char*>clo.base)
    ind /= clo.elsize
    cdef char * rptr = <char*>radix
    cdef uint64_t value
    for i in range(clo.radixarray_u8.shape[1]):
        value = clo.radixarray_u8[ind, i]
        memcpy(rptr, &value, 8)
        rptr += 8

cdef void myradix_i8(void * ptr, void * radix, void * arg):
    cdef MyClosure clo = <MyClosure> arg
    cdef numpy.intp_t ind
    cdef int i
    ind = (<char*>ptr - <char*>clo.base)
    ind /= clo.elsize
    cdef char * rptr = <char*>radix
    cdef int64_t value
    for i in range(clo.radixarray_i8.shape[1]):
        value = clo.radixarray_i8[ind, i]
        value += <int64_t>-9223372036854775808
        memcpy(rptr, &value, 8)
        rptr += 8

cdef void myradix_u4(void * ptr, void * radix, void * arg):
    cdef MyClosure clo = <MyClosure> arg
    cdef numpy.intp_t ind
    cdef int i
    ind = (<char*>ptr - <char*>clo.base)
    ind /= clo.elsize
    cdef char * rptr = <char*>radix
    cdef uint64_t value
    for i in range(clo.radixarray_u4.shape[1]):
        value = clo.radixarray_u4[ind, i]
        memcpy(rptr, &value, 8)
        rptr += 8

cdef void myradix_i4(void * ptr, void * radix, void * arg):
    cdef MyClosure clo = <MyClosure> arg
    cdef numpy.intp_t ind
    cdef int i
    ind = (<char*>ptr - <char*>clo.base)
    ind /= clo.elsize
    cdef char * rptr = <char*>radix
    cdef int64_t value
    for i in range(clo.radixarray_i4.shape[1]):
        value = clo.radixarray_i4[ind, i]
        value += <int64_t>-9223372036854775808
        memcpy(rptr, &value, 8)
        rptr += 8

def sort(numpy.ndarray data, orderby=None, comm=None):
    """ 
        Parallel sort of distributed data set `data' over MPI Communicator `comm', 
        ordered by key given in 'orderby'. 

        data[orderby] must be of dtype `uint64'. 
        data[orderby] can be 2d, in which case the latter elements in a row has 
        more significance.
        
        data must be C_contiguous numpy arrays,
        
        if orderby is None, use data itself.
    """
    cdef MyClosure clo = MyClosure(data, orderby)
    cdef MPI.MPI_Comm mpicomm
    if not data.flags['C_CONTIGUOUS']:
        raise ValueError("data must be C_CONTIGUOUS")

    if comm is None:
        mpicomm = MPI.MPI_COMM_WORLD
    else:
        if isinstance(comm, pyMPI.Comm):
            if hasattr(pyMPI, '_addressof'):
                mpicomm = (<MPI.MPI_Comm*> (<numpy.intp_t>
                        pyMPI._addressof(comm))) [0]
            else:
                raise ValueError("only comm=None is supported, "
                        + " update mpi4py to a version with MPI._addressof")
        else:
            raise ValueError("only MPI.Comm objects are supported")

    radix_sort_mpi(data.data, data.shape[0], data.strides[0], clo.myradix, 
            clo.radix_length * 8, <void*>clo, mpicomm)

