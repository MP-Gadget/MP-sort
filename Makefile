MPICC=mpicc

all: libradixsort.a libradixsort-omp.a libradixsort-mpi.a
clean:
	rm *.o *.a	
tests: main main-mpi bench-mpi
main: main.c libradixsort-omp.a libradixsort.a
	$(CC) -o main $^
main-mpi: main-mpi.c libradixsort-mpi.a libradixsort.a
	$(MPICC) -o main-mpi $^
bench-mpi: bench-mpi.c libradixsort-mpi.a libradixsort.a
	$(MPICC) -o bench-mpi $^

libradixsort.a: radixsort.c
	$(CC) -c -o radixsort.o radixsort.c
	ar r libradixsort.a radixsort.o
	ranlib libradixsort.a

libradixsort-omp.a: radixsort-omp.c
	$(CC) -c -o radixsort-omp.o radixsort-omp.c
	ar r libradixsort-omp.a radixsort-omp.o
	ranlib libradixsort-omp.a

libradixsort-mpi.a: radixsort-mpi.c
	$(CC) -c -o radixsort-mpi.o radixsort-mpi.c
	ar r libradixsort-mpi.a radixsort-mpi.o
	ranlib libradixsort-mpi.a

