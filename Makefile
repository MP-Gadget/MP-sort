MPICC=mpicc

all:main main-mpi
main: main.c radixsort.c radixsort-omp.c
	$(CC) -o main main.c radixsort.c radixsort-omp.c
main-mpi: main-mpi.c radixsort.c radixsort-mpi.c
	$(MPICC) -o main-mpi main-mpi.c radixsort.c radixsort-mpi.c
