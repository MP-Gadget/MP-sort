main: main.c radixsort.c radixsort-omp.c
	gcc -o main -g -O3 main.c radixsort.c radixsort-omp.c -fopenmp
