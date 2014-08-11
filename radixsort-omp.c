#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#include "internal.h"

#include "_bsearch.c"
#include "radixsort.h"

struct crompstruct {
    char * P0;
    char * P;
    char * Pleft;
    char * Pright;
    char * Pmax;
    char * Pmin;
    int * stable;
    int * narrow;
    ptrdiff_t * C; /* expected counts */
    ptrdiff_t * CLT; /* counts of less than P */
    ptrdiff_t * CLE; /* counts of less than or equal to P */
    ptrdiff_t * CLTall; /* counts of less than P */
    ptrdiff_t * CLEall; /* counts of less than or equal to P */
    ptrdiff_t * Call; /* counts of less than or equal to P */
};


/* OPENMP version of radix sort; 
 * this is truely parallel; 
 * but it is usually slower than
 * simple radix sort if the number of processor is small.
 *
 * some benchmarks on Coma at CMU shows best performance is at 16
 * CPUs; still faster than serial version with 8 CPUs.
 * comparable with qsort (non-radix) at 8 CPUs.
 * 
 * the coding is more of a prototype of the MPI radix sort;
 * it is thus very poorly written in the standards of an OPENMP program; 
 * */

static void _setup_radix_sort_omp(struct crompstruct * o, struct crstruct * d);
static void _cleanup_radix_sort_omp(struct crompstruct * o, struct crstruct * d);

static void radix_sort_omp_single(void * base, size_t nmemb, 
        struct crstruct * d, struct crompstruct * o);

void radix_sort_omp(void * base, size_t nmemb, size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg) {

    struct crstruct d;
    struct crompstruct o;
    _setup_radix_sort(&d, size, radix, rsize, arg);

    _setup_radix_sort_omp(&o, &d);

    /* 
     * first solve for P such that CLT[i] <= C <= CLE[i] 
     * 
     * Then calculate a communication layout.
     *
     * Then alltoall.
     * Then local sort again
     * */

#pragma omp parallel
    {
        int NTask = omp_get_num_threads();
        o.Pmax = o.P + NTask * d.rsize;
#pragma omp master 
        {
            int i;
            o.C[0] = 0;
            for(i = 0; i < NTask; i ++) {
            /* how many items are desired per thread */
                o.C[i + 1] = nmemb * (i + 1) / NTask;
            }
        }
#pragma omp barrier
        radix_sort_omp_single (base, nmemb, &d, &o);
    }

    _cleanup_radix_sort_omp(&o, &d);
}

static void _setup_radix_sort_omp(struct crompstruct * o, struct crstruct * d) {
    int NTaskMax = omp_get_max_threads();
    o->stable = calloc(NTaskMax, sizeof(int));
    o->narrow = calloc(NTaskMax, sizeof(int));

    o->P0 = calloc(NTaskMax + 2, d->rsize);
    o->Pleft = calloc(NTaskMax, d->rsize);
    o->Pright = calloc(NTaskMax, d->rsize);
    o->P = o->P0 + d->rsize;

    int NTaskMax1 = NTaskMax + 1;
    size_t NTaskMax12 = (NTaskMax + 1) * (NTaskMax + 1);

    /* following variables are used in counting index by ThisTask + 1 */
    o->CLTall = calloc(NTaskMax12, sizeof(ptrdiff_t));
    o->CLEall = calloc(NTaskMax12, sizeof(ptrdiff_t));
    o->Call = calloc(NTaskMax12, sizeof(ptrdiff_t));

    o->C = calloc(NTaskMax1, sizeof(ptrdiff_t)); /* expected counts */
    o->CLT = calloc(NTaskMax1, sizeof(ptrdiff_t)); /* counts of less than P */
    o->CLE = calloc(NTaskMax1, sizeof(ptrdiff_t)); /* counts of less than or equal to P */

    o->Pmin = o->P0;
    memset(o->Pmin, -1, d->rsize);
}
static void _cleanup_radix_sort_omp(struct crompstruct * o, struct crstruct * d) {
    free(o->CLE);
    free(o->CLT);
    free(o->C);
    free(o->Call);
    free(o->CLEall);
    free(o->CLTall);
    free(o->P0);
    free(o->Pleft);
    free(o->Pright);
    free(o->stable);
    free(o->narrow);

}

static void _count_once_mine(char * P, void * mybase, size_t mynmemb, 
        ptrdiff_t * myCLT, ptrdiff_t * myCLE,
        struct crstruct * d, struct crompstruct * o) {
    int NTask = omp_get_num_threads();
    int it;

    myCLT[0] = 0;
    myCLE[0] = 0;
    for(it = 0; it < NTask - 1; it ++) {
        myCLT[it + 1] = _bsearch_last_lt(P + it * d->rsize, mybase, mynmemb, d) + 1;
        myCLE[it + 1] = _bsearch_last_le(P + it * d->rsize, mybase, mynmemb, d) + 1;
    }
    myCLT[it + 1] = mynmemb;
    myCLE[it + 1] = mynmemb;
}

static void radix_sort_omp_single_iteration(char * mybase, size_t mynmemb, 
        struct crstruct * d, 
        struct crompstruct * o) {
    int NTask = omp_get_num_threads();
    int ThisTask = omp_get_thread_num();

    memcpy(&o->Pleft[ThisTask * d->rsize], o->Pmin, d->rsize);
    memcpy(&o->Pright[ThisTask * d->rsize], o->Pmax, d->rsize);

    ptrdiff_t myCLT[NTask + 1]; /* counts of less than P */
    ptrdiff_t myCLE[NTask + 1]; /* counts of less than or equal to P */

    int iter = 0;
    int i;

    int done = 0;

    for(i = 0; i < NTask; i ++) {
        o->narrow[i] = 0;
        o->stable[i] = 0;
    }

    while(!done) {

#pragma omp master
        {
            for(i = 0; i < NTask - 1; i ++) {
                if(o->narrow[i]) {
                    /* The last iteration, test Pright directly */
                    memcpy(&o->P[i * d->rsize],
                        &o->Pright[i * d->rsize], 
                        d->rsize);
                    o->stable[i] = 1;
                } else {
                    /* ordinary iteration */
                    d->bisect(&o->P[i * d->rsize], 
                            &o->Pleft[i * d->rsize], 
                            &o->Pright[i * d->rsize], d->rsize);
                    /* in case the bisect can't move P beyond left,
                     * the range is too small, so we set flag narrow, 
                     * and next iteration we will directly test Pright */
                    if(d->compar(&o->P[i * d->rsize], 
                        &o->Pleft[i * d->rsize], d->rsize) == 0) {
                        o->narrow[i] = 1; 
                    }
                }
#if 0
                printf("bisect %d %d %u %u %u\n", iter, i, *(int*) &o->P[i * d->rsize], 
                        *(int*) &o->Pleft[i * d->rsize], 
                        *(int*) &o->Pright[i * d->rsize]);
#endif
            }
            for(i = 0; i < NTask + 1; i ++) {
                o->CLT[i] = 0;
                o->CLE[i] = 0;
            }
        }

#pragma omp barrier
        iter ++;

        _count_once_mine(o->P, mybase, mynmemb, myCLT, myCLE, d, o);

        /* reduce the counts */

#pragma omp barrier

        for(i = 0; i < NTask + 1; i ++) {
#pragma omp atomic
            o->CLT[i] += myCLT[i];
#pragma omp atomic
            o->CLE[i] += myCLE[i];
        }

#pragma omp barrier

#pragma omp master 
        {
            for(i = 0; i < NTask - 1; i ++) {
                if(o->stable[i]) continue;
                if( o->CLT[i + 1] < o->C[i + 1] && o->C[i + 1] <= o->CLE[i + 1]) {
                    memcpy(&o->Pright[i * d->rsize], 
                           &o->P[i * d->rsize], d->rsize);
                } else {
                    if(o->CLT[i + 1] >= o->C[i + 1]) {
                        /* P[i] is too big */
                        memcpy(&o->Pright[i * d->rsize], &o->P[i * d->rsize], d->rsize);
                    } else {
                        /* P[i] is too small */
                        memcpy(&o->Pleft[i * d->rsize], &o->P[i * d->rsize], d->rsize);
                    }
                }
            }

#if 0
            for(i = 0; i < NTask + 1; i ++) {
                printf("counts %d %d LT %ld C %ld LE %ld\n", iter, i, o->CLT[i], o->C[i], o->CLE[i]);
            }
#endif
        }
#pragma omp barrier

        done = 1;
        int i;
        for(i = 0; i < NTask - 1; i ++) {
            if(!o->stable[i]) {
                done = 0;
            }
        }
#pragma omp barrier
    }

}
static void _find_split_points(struct crompstruct * o) {
    int NTask = omp_get_num_threads();
    int NTask1 = NTask + 1;
#pragma omp barrier
#pragma omp master
    {
        int i, j;
        for(i = 0; i < NTask + 1; i ++) {
            for(j = 0; j < NTask; j ++) {
                o->Call[j * NTask1 + i] = o->CLTall[j * NTask1 + i];
            }
        }
        for(i = 0; i < NTask; i ++) {
            ptrdiff_t sure = 0;
            /* solve*/
            for(j = 0; j < NTask; j ++) {
                sure += o->Call[j * NTask1 + i + 1] - o->Call[j * NTask1 + i];
            }
            ptrdiff_t diff = o->C[i + 1] - o->C[i] - sure;
            for(j = 0; j < NTask; j ++) {
                if(diff < 0) abort();
                if(diff == 0) break;
                ptrdiff_t supply = o->CLEall[j * NTask1 + i + 1] - o->Call[j * NTask1 + i + 1];
                if(supply <= diff) {
                    o->Call[j * NTask1 + i + 1] += supply;
                    diff -= supply;
                } else {
                    o->Call[j * NTask1 + i + 1] += diff;
                    diff = 0;
                }
            }
        }

#if 0
        for(i = 0; i < NTask; i ++) {
            for(j = 0; j < NTask + 1; j ++) {
                printf("%d %d %d, ", 
                        o->CLTall[i * NTask1 + j], 
                        o->Call[i * NTask1 + j], 
                        o->CLEall[i * NTask1 + j]);
            }
            printf("\n");
        }
#endif
    }
#pragma omp barrier

}
static void radix_sort_omp_single(void * base, size_t nmemb, 
        struct crstruct * d, struct crompstruct * o) {
    int NTask = omp_get_num_threads();
    int ThisTask = omp_get_thread_num();

    double t0 = omp_get_wtime();

    /* distribute the array and sort the local array */
    char * mybase = (char*) base + nmemb * ThisTask / NTask * d->size;
    size_t mynmemb = nmemb * (ThisTask + 1)/ NTask - nmemb * (ThisTask) / NTask;


    radix_sort(mybase, mynmemb, d->size, d->radix, d->rsize, d->arg);

    /* find the max radix and min radix of all */
    if(mynmemb > 0) {
        char * myPmax[d->rsize];
        char * myPmin[d->rsize];
        d->radix(mybase + (mynmemb - 1) * d->size, myPmax, d->arg);
        d->radix(mybase, myPmin, d->arg);
#pragma omp critical
        {
            if(d->compar(myPmax, o->Pmax, d->rsize) > 0) {
                memcpy(o->Pmax, myPmax, d->rsize);
            }
            if(d->compar(myPmin, o->Pmin, d->rsize) < 0) {
                memcpy(o->Pmin, myPmin, d->rsize);
            }
        }
    }
#pragma omp barrier

    double t1 = omp_get_wtime();
    printf("Initial sort took %g\n", t1 - t0);

    
    /* now do the radix counting iterations */ 

    radix_sort_omp_single_iteration(mybase, mynmemb, d, o);

    double t2 = omp_get_wtime();
    printf("counting took %g\n", t2 - t1);

#pragma omp barrier
#if 0
#pragma omp master 
    {
        printf("AfterIteration: split , CLT, C, CLE\n");
        int i;
        for(i = 0; i < NTask + 1; i ++) {
            printf("%d %ld %ld %ld\n", i, o->CLT[i], o->C[i], o->CLE[i]);
        }
    }
#endif
    ptrdiff_t myCLT[NTask + 1]; /* counts of less than P */
    ptrdiff_t myCLE[NTask + 1]; /* counts of less than or equal to P */

    _count_once_mine(o->P, mybase, mynmemb, myCLT, myCLE, d, o);

    /* allgather */
    int NTask1 = NTask + 1;
    int i;
    for(i = 0; i < NTask + 1; i ++) {
        o->CLTall[ThisTask * NTask1 + i] = myCLT[i];
        o->CLEall[ThisTask * NTask1 + i] = myCLE[i];
    }

    /* indexing is
     * Call[Sender * NTask1 + Recver] */
    /* here we know:
     * o->CLTall, o->CLEall
     * The first NTask - 1 items in CLT and CLE gives the bounds of
     * split points  ( CLT <= split < CLE
     * */

    /* find split points in O->Call*/
    _find_split_points(o);

    double t3 = omp_get_wtime();
    printf("find split took %g\n", t3 - t2);

#pragma omp barrier


    /* exchange data */
    /* */
    char * buffer = malloc(d->size * mynmemb);

#if 0
#pragma omp critical 
    {
        printf("%d contains %d items ", ThisTask, mynmemb);
        int k;
        int * ptr = mybase;
        for(k = 0; k < mynmemb; k ++) {
            printf("%d ", ((int *) ptr)[k]);
        }
        printf("\n");
    }
#pragma omp barrier
#endif
    char * recv = buffer;
    for(i = 0; i < NTask; i ++) {
        char * sendbuf = (char*) base + d->size * (i * nmemb / NTask);
        size_t size = (o->Call[i * NTask1 + ThisTask + 1]
                - o->Call[i * NTask1 + ThisTask]) * d->size;
        char * ptr = &sendbuf[d->size * o->Call[i * NTask1 + ThisTask]];

        memcpy(recv, ptr, size);
        recv += size;
    } 

#pragma omp barrier
    memcpy(mybase, buffer, mynmemb * d->size);
    free(buffer);

#if 0
#pragma omp critical 
    {
        printf("%d after exchange %d items ", ThisTask, mynmemb);
        int k;
        int * ptr = mybase;
        for(k = 0; k < mynmemb; k ++) {
            printf("%d ", ((int *) ptr)[k]);
        }
        printf("\n");
    }
#pragma omp barrier
#endif

    double t4 = omp_get_wtime();
    printf("exchange took %g\n", t4 - t3);

    radix_sort(mybase, mynmemb, d->size, d->radix, d->rsize, d->arg);

    double t5 = omp_get_wtime();
    printf("final sort %g\n", t5 - t4);

#pragma omp barrier
}

