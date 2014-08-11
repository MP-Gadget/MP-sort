main: main.c radixsort.c radixsort-omp.c
	$(CC) -o main main.c radixsort.c radixsort-omp.c -fopenmp
