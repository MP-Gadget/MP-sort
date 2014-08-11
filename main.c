#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#include "radixsort.h"

static void radix_int(const void * ptr, void * radix, void * arg) {
    *(int*)radix = *(const int*) ptr;
}
static int compar_int(const void * p1, const void * p2) {
    const unsigned int * i1 = p1, *i2 = p2;
    return (*i1 > *i2) - (*i1 < *i2);
}

int main() {
    int i;
    srand(9999);
#define NUMITEMS 10000000
    int * data1 = malloc(sizeof(int) * NUMITEMS);
    int * data2 = malloc(sizeof(int) * NUMITEMS);
    int * data3 = malloc(sizeof(int) * NUMITEMS);
    for(i = 0; i < NUMITEMS; i ++) {
        data1[i] = random() % 10000;
        data2[i] = data1[i];
        data3[i] = data1[i];
    }

    {
        double t0 = 1.0 * clock() / CLOCKS_PER_SEC;
        radix_sort_omp(data2, NUMITEMS, sizeof(int),
                radix_int, sizeof(int),
                NULL);
        double t1 = 1.0 * clock() / CLOCKS_PER_SEC;
        printf("time spent omp: %g\n", t1 - t0);
    }

    {
        double t0 = 1.0 * clock() / CLOCKS_PER_SEC;
        qsort(data3, NUMITEMS, sizeof(int), compar_int);
        double t1 = 1.0 * clock() / CLOCKS_PER_SEC;
        printf("time spent qsort: %g\n", t1 - t0);
    }
    {
        double t0 = 1.0 * clock() / CLOCKS_PER_SEC;
        radix_sort(data1, NUMITEMS, sizeof(int),
                radix_int, sizeof(int),
                NULL);
        printf("max is %u\n", data1[NUMITEMS - 1]);
        double t1 = 1.0 * clock() / CLOCKS_PER_SEC;
        printf("time spent radix: %g\n", t1 - t0);
    }
    for(i = 0; i < NUMITEMS; i ++) {
        if(data1[i] != data2[i]) abort();
    }
    return 0;
}
