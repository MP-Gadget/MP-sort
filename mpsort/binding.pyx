#cython: embedsignature=True
cimport numpy
cimport libmpi as MPI
from libc.stddef cimport ptrdiff_t
from libc.stdint cimport uint64_t, int64_t, uint32_t, int32_t
from libc.string cimport memcpy
from libc.stdlib cimport abort
import numpy
from mpi4py import MPI as pyMPI

cdef extern from "radixsort.c":
    pass

cdef extern from "mpsort-mpi.c":
    int MPSORT_ENABLE_SPARSE_ALLTOALLV
    int MPSORT_DISABLE_IALLREDUCE
    int MPSORT_DISABLE_GATHER_SORT
    int MPSORT_REQUIRE_GATHER_SORT
    int MPSORT_REQUIRE_SPARSE_ALLTOALLV

    void mpsort_mpi_set_options(int options)
    void mpsort_mpi_unset_options(int options)
    void mpsort_mpi_newarray(void * base, size_t nmemb, 
            void * outbase, size_t outnmemb,
            size_t size,
            void (*radix)(void * ptr, void * radix, void * arg),
            size_t rsize, 
            void * arg, MPI.MPI_Comm comm)

cdef struct MyClosure:
    int elsize
    void (*myradix)(const void * ptr, void * radix, void * arg) nogil
    ptrdiff_t radix_offset
    int radix_nmemb

cdef closure_init(MyClosure * self, numpy.dtype dtype, radixkey):
    cdef numpy.dtype radixdtype

    self.elsize = dtype.itemsize

    if radixkey is not None:
        radixdtype, self.radix_offset = dtype.fields[radixkey]
    else:
        radixdtype, self.radix_offset = dtype, 0

    if len(radixdtype.shape) == 0:
        self.radix_nmemb = 1
    elif len(radixdtype.shape) == 1:
        self.radix_nmemb = radixdtype.shape[0]
    else:
        raise ValueError("data[%s] is not 1d nor 2d %s" % (radixkey))

    #print 'radix offset =', self.radix_offset
    #print 'radix nmemb =', self.radix_nmemb
    #print 'radix dtype.shape = ', radixdtype.shape
    if radixdtype.base == numpy.dtype('u8'):
        self.myradix = myradix_u8
    elif radixdtype.base == numpy.dtype('i8'):
        self.myradix = myradix_i8
    elif radixdtype.base == numpy.dtype('u4'):
        self.myradix = myradix_u4
    elif radixdtype.base == numpy.dtype('i4'):
        self.myradix = myradix_i4
    else:
        raise TypeError("data[%s] is not u8 or i8" % (radixkey))

cdef void myradix_u8(const void * ptr, void * radix, void * arg) nogil:
    cdef MyClosure *clo = <MyClosure*> arg
    cdef char * rptr = <char*>radix
    cdef char * cptr = <char*> ptr
    cdef uint64_t value
    for i in range(clo.radix_nmemb):
        value = (<uint64_t *> (cptr + clo.radix_offset))[i]
        memcpy(rptr, &value, 8)
        rptr += 8

cdef void myradix_i8(const void * ptr, void * radix, void * arg) nogil:
    cdef MyClosure *clo = <MyClosure*> arg
    cdef char * rptr = <char*>radix
    cdef char * cptr = <char*> ptr
    cdef uint64_t value
    for i in range(clo.radix_nmemb):
        value = (<int64_t *> (cptr + clo.radix_offset))[i]
        value += <uint64_t> 9223372036854775808uL
        memcpy(rptr, &value, 8)
        rptr += 8

cdef void myradix_u4(const void * ptr, void * radix, void * arg) nogil:
    cdef MyClosure *clo = <MyClosure*> arg
    cdef char * rptr = <char*>radix
    cdef char * cptr = <char*> ptr
    cdef uint64_t value
    for i in range(clo.radix_nmemb):
        value = (<uint32_t *> (cptr + clo.radix_offset))[i]
        memcpy(rptr, &value, 8)
        rptr += 8

cdef void myradix_i4(const void * ptr, void * radix, void * arg) nogil:
    cdef MyClosure *clo = <MyClosure*> arg
    cdef char * rptr = <char*>radix
    cdef char * cptr = <char*> ptr
    cdef uint64_t value
    for i in range(clo.radix_nmemb):
        value = (<int32_t *> (cptr + clo.radix_offset))[i]
        value += <uint64_t> 9223372036854775808uL
        memcpy(rptr, &value, 8)
        rptr += 8

def sort(numpy.ndarray data, orderby=None, numpy.ndarray out=None, comm=None, tuning=[]):
    """
        Parallel sort of distributed data set `data' over MPI Communicator `comm',
        ordered by key given in 'orderby'.

        Parameters
        ----------
        data : numpy.ndarray
            the input data; must be C_contiguous numpy arrays,

        orderby : string or indices

            data[orderby] must be of integer types.
            data[orderby] can be 2d, in which case the latter elements in a row has
            more significance.

            if orderby is None, use data itself.

        out : numpy.ndarray or None
            the output array; if None, inplace

        comm : MPIComm or None
            the communicaotr, None for COMM_WORLD

        tuning: list of strings
            'ENABLE_SPARSE_ALLTOALLV'
            'DISABLE_IALLREDUCE'
            'DISABLE_GATHER_SORT'
            'REQUIRE_GATHER_SORT'
            'REQUIRE_SPARSE_ALLTOALLV'
    """
    cdef MyClosure clo
    cdef MPI.MPI_Comm mpicomm

    # assert you can access the orderby columns.
    key = data[orderby]

    if not data.flags['C_CONTIGUOUS']:
        raise ValueError("data must be C_CONTIGUOUS")

    if out is None:
        out = data

    if not out.flags['C_CONTIGUOUS']:
        raise ValueError("out must be C_CONTIGUOUS")

    if comm is None:
        comm = pyMPI.COMM_WORLD
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

    Ntot = comm.allreduce(len(data))
    Ntotout = comm.allreduce(len(out))

    if Ntot != Ntotout:
        raise ValueError("total size of array changed %d != %d" % (Ntot, Ntotout))

    if data.dtype.itemsize != out.dtype.itemsize:
        raise ValueError("item size mismatch")

    closure_init(&clo, data.dtype, orderby)

    # hope that GIL ensures nobody will mess with the options

    mpsort_mpi_unset_options(-1)

    if 'ENABLE_SPARSE_ALLTOALLV' in tuning:
        mpsort_mpi_set_options(MPSORT_ENABLE_SPARSE_ALLTOALLV)
    if 'DISABLE_IALLREDUCE' in tuning:
        mpsort_mpi_set_options(MPSORT_DISABLE_IALLREDUCE)
    if 'DISABLE_GATHER_SORT' in tuning:
        mpsort_mpi_set_options(MPSORT_DISABLE_GATHER_SORT)
    if 'REQUIRE_GATHER_SORT' in tuning:
        mpsort_mpi_set_options(MPSORT_REQUIRE_GATHER_SORT)
    if 'REQUIRE_SPARSE_ALLTOALLV' in tuning:
        mpsort_mpi_set_options(MPSORT_REQUIRE_SPARSE_ALLTOALLV)

    mpsort_mpi_newarray(data.data, len(data),
            out.data, len(out),
            data.dtype.itemsize, clo.myradix,
            clo.radix_nmemb * 8, <void*>&clo, mpicomm)

