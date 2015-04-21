#cython: embedsignature=True
cimport numpy
cimport libmpi as MPI
from libc.stddef cimport ptrdiff_t
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
    cdef numpy.uint64_t [:, :] radixarray
    def __init__(self, numpy.ndarray dataarray, radixkey):
        self.base = dataarray.data
        self.elsize = dataarray.strides[0]
        if radixkey is not None:
            radixarray = dataarray[radixkey]
        else:
            radixarray = dataarray

        if len(radixarray.shape) == 1:
            self.radixarray = radixarray.reshape(-1, 1)
        elif len(radixarray.shape) == 2:
            self.radixarray = radixarray
        else:
            raise ValueError("data[%s] is not 1d nor 2d %s" % (radixkey,
                str(dataarray[radixkey].shape)))

cdef void myradix(void * ptr, void * radix, void * arg):
    cdef MyClosure clo = <MyClosure> arg
    cdef numpy.intp_t ind
    cdef int i
    ind = (<char*>ptr - <char*>clo.base)
    ind /= clo.elsize
    cdef char * rptr = <char*>radix
    for i in range(clo.radixarray.shape[1]):
        memcpy(rptr, &clo.radixarray[ind, i], 8)
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

    radix_sort_mpi(data.data, data.shape[0], data.strides[0], myradix, 
            clo.radixarray.shape[1] * 8, <void*>clo, mpicomm)

