MP-sort
=======

A Massively Parallel Sorting Library. The library implements a histogram
sort. The scaling of MP-sort up to 160,000 MPI ranks has been studied by 
[1]_. MP-Sort is the sorting module in BlueTides Simulation [2]_.

Our vision is that MP-sort will significantly ease
BigData analysis on traditional HPC platforms with MPI. 

Based on this vision, we provide 

- a C interface that can be easily integrated
  into BigData producers, such as large scale simulation applications; and

- a Python interface that can be easily integrated 
  into BigData consumers, such as parallel Python scripts 
  that queries simulation data.

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

    mpsort.sort(localdata, orderby=None)

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
    
        
.. [1] Feng, Y., Straka, M., Di Matteo, T., Croft, R., MP-Sort: Sorting for a Cosmological Simulation on BlueWaters, Cray User Group 2015
.. [2] Feng et. al, BlueTides: First galaxies and reionization, Monthly Notices of the Royal Astronomical Society, 2015, submitted

