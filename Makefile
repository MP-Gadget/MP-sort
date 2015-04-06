CC=cc
MPICC=mpicc
PREFIX=/usr

all: libradixsort.a libradixsort-mpi.a

install: libradixsort.a libradixsort-mpi.a
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install libradixsort.a $(PREFIX)/lib/libradixsort.a
	install libradixsort-mpi.a $(PREFIX)/lib/libradixsort-mpi.a
	install radixsort.h $(PREFIX)/include/radixsort-mpi.h

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
	$(MPICC) -c -o radixsort-mpi.o radixsort-mpi.c
	ar r libradixsort-mpi.a radixsort-mpi.o
	ranlib libradixsort-mpi.a

