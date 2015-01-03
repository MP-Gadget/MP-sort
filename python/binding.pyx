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
    cdef numpy.uint64_t [::1] radixarray
    def __init__(self, numpy.ndarray dataarray, radixarray):
        self.base = dataarray.data
        self.elsize = dataarray.strides[0]
        self.radixarray = radixarray

cdef void myradix(void * ptr, void * radix, void * arg):
    cdef MyClosure clo = <MyClosure> arg
    cdef numpy.intp_t ind
    ind = (<char*>ptr - <char*>clo.base)
    ind /= clo.elsize
    memcpy(radix, &clo.radixarray[ind], 8)

def sort(numpy.ndarray data, radix, comm=None):
    """ data and radix must be C_contiguous numpy arrays,

        radix must be of uint64  """
    cdef MyClosure clo = MyClosure(data, radix)
    cdef MPI.MPI_Comm mpicomm
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

    radix_sort_mpi(data.data, data.shape[0], data.strides[0], myradix, 8, <void*>clo, mpicomm)

