#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include "radixsort.h"

static double wtime() {
    struct timespec t1;
    clock_gettime(CLOCK_REALTIME, &t1);
    return (double)((t1.tv_sec+t1.tv_nsec*1e-9));
}

static void radix_int(const void * ptr, void * radix, void * arg) {
    *(int*)radix = *(const int*) ptr;
}
static int compar_int(const void * p1, const void * p2) {
    const unsigned int * i1 = p1, *i2 = p2;
    return (*i1 > *i2) - (*i1 < *i2);
}

int main(int argc, char * argv[]) {
    int i;
    srand(9999);

    MPI_Init(&argc, &argv);

    int ThisTask;
    int NTask;

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(argc != 2) {
        printf("./main [number of items]\n");
        return 1;
    }

    size_t NUMITEMS = atoi(argv[1]);
    int * data1 = malloc(sizeof(int) * NUMITEMS);
    int * data2 = malloc(sizeof(int) * NUMITEMS);

    int * mydata1 = data1 + ThisTask * NUMITEMS / NTask;
    int * mydata2 = data2 + ThisTask * NUMITEMS / NTask;
    size_t mynumitems = (ThisTask + 1 ) * NUMITEMS / NTask - ThisTask * NUMITEMS / NTask;

    for(i = 0; i < NUMITEMS; i ++) {
        data1[i] = random() % 10000;
        data2[i] = data1[i];
    }

    {
        double t0 = wtime();
        radix_sort_mpi(mydata1, mynumitems, sizeof(int),
                radix_int, sizeof(int),
                NULL, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = wtime();
        if(ThisTask == 0)  {
            printf("time spent omp: %g\n", t1 - t0);
            radix_sort_mpi_report_last_run();
        }
    }

    {
        double t0 = wtime();
        qsort(data2, NUMITEMS, sizeof(int), compar_int);
        double t1 = wtime();
        if(ThisTask == 0) {
            printf("time spent qsort: %g\n", t1 - t0);
        }
    }

    for(i = 0; i < mynumitems; i ++) {
        if(mydata1[i] != mydata2[i]) abort();
    }
    free(data1);
    free(data2);
    MPI_Finalize();
    return 0;
}
