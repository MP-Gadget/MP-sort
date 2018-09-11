MP-sort
=======


.. image:: https://travis-ci.org/rainwoodman/MP-sort.svg?branch=master
       :target: https://travis-ci.org/rainwoodman/MP-sort

DOI:
.. image:: https://zenodo.org/badge/22829423.svg
   :target: https://zenodo.org/badge/latestdoi/22829423
   
A Massively Parallel Sorting Library. The library implements a histogram
sort. The scaling of MP-sort up to 160,000 MPI ranks has been studied by 
[1]_. MP-Sort is the sorting module in BlueTides Simulation [2]_.

MP-sort can be very useful for
BigData simulation and analysis on traditional HPC platforms with MPI. 

We provide 

- a C interface that can be easily integrated
  into data producers, such as large scale simulation applications; and

- a Python interface that can be easily integrated 
  into data consumers, such as parallel Python scripts 
  that query simulation data. 

A real world Python example is https://github.com/bccp/nbodykit/blob/master/ichalo.py , where mpsort is used twice to match up properties of particles cross two different snapshots, to establish the motion of Dark Matter halos cross cosmic times.

Install
-------

Use the Makefile to build / install the .a targets, and link against 

.. code:: bash

    -lradixsort -lmpsort-mpi

Makefile supports overriding :code:`CC`, :code:`MPICC` and :code:`CFLAGS`

The header to include is

.. code:: c
    
    #include <mpsort.h>


The python binding can be installed with

.. code:: bash

    python setup.py install [--user]

The same Makefile overrides are supported, if the environment variables are set.

Usage: C
--------

The basic C interface is:

.. code:: c

    void mpsort_mpi(void * base, size_t nmemb, size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg, MPI_Comm comm);

    /*
    Parameters
    ----------
    base :
        base pointer of the local data
    nmemb :
        number of items in the local data
    elsize :
        size of an item in the local data
    radix (ptr, radix, arg):
        the function calculates the radix of an item at ptr;
        the raxis shall be stored at memory location pointed to radix
    rsize :
        the size of radix array in bytes; only the behavior for 8 is well-defined:
        the radix is interpreted as uint64.
    arg   :
        argument to pass into radix()
    comm  :
        the MPI communicator for the sort. 

    */

Usage: Python
-------------

The basic Python interface is:

.. code:: python
    
    import mpsort

    mpsort.sort(localdata, orderby=None, comm=None, tuning=[])

    """
    Sort an distributed array in place.

    Parameters
    ----------
    localdata : array_like
        local data, must be C_CONTIGUOUS, and of a struct-dtype.
        for example, :code:`localdata = numpy.empty(10, dtype=[('key', 'i4'), ('value', 'f4')])`.
    orderby : scalar
        the field to be sorted by. The field must be of an integral type. 'i4', 'i8', 'u4', 'u8'.

    """

Tuning
------

For runs with very large number of ranks, we may experience slow down due to the backend selecting a conservative `MPI_Allreducev` implementation. For Cray MPI, the following environment helps:

.. code::

    export MPICH_ALLREDUCE_BLK_SIZE=$((4096*1024*2))

There are also flags controlling the algorithm used by `MPI_Alltoallv`.

If the communication is very sparse (e.g. mostly sorted data). Then using a sparse algorithm based on
non-blocking send / recv may provide better performance; some MPI implementations do not automatically switch
the algorithm. We can allow mpsort to automatically select a sparse algorithm by setting
the environment `MPSORT_ENABLE_SPARSE_ALLTOALLV`, calling `mpsort_mpi_set_option(MPSORT_ENABLE_SPARSE_ALLTOALLV)`, or passing `'ENABLE_SPARSE_ALLTOALLV'` to the tuning
argument of the python interface.

On MPI-3, MP-Sort will use non-blocking `MPI_Iallreduce` to build the histogram. This may not always be desirable. Set `MPSORT_DISABLE_IALLREDUCE` to avoid this.


.. [1] Feng, Y., Straka, M., Di Matteo, T., Croft, R., MP-Sort: Sorting for a Cosmological Simulation on BlueWaters, Cray User Group 2015
.. [2] Feng et. al, BlueTides: First galaxies and reionization, Monthly Notices of the Royal Astronomical Society, 2015, submitted


