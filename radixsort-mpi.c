
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <mpi.h>

#include "internal.h"

#include "internal-parallel.h"

/* mpi version of radix sort; 
 *
 * each caller provides the distributed array and number of items.
 * the sorted array is returned to the original array pointed to by 
 * mybase. (AKA no rebalancing is done.)
 *
 * NOTE: may need an api to return a balanced array!
 *
 * uses the same amount of temporary storage space for communication
 * and local sort. (this will be allocated via malloc)
 *
 *
 * */

static MPI_Datatype MPI_TYPE_PTRDIFF = 0;
struct crmpistruct {
    MPI_Datatype MPI_TYPE_RADIX;
    MPI_Comm comm;
    int NTask;
    int ThisTask;
};

static void _setup_radix_sort_mpi(struct crmpistruct * o, struct crstruct * d, MPI_Comm comm) {
    o->comm = comm;
    MPI_Comm_size(comm, &o->NTask);
    MPI_Comm_rank(comm, &o->ThisTask);

    if(MPI_TYPE_PTRDIFF == 0) {
        if(sizeof(ptrdiff_t) == sizeof(long long)) {
            MPI_TYPE_PTRDIFF = MPI_LONG_LONG;
        }
        if(sizeof(ptrdiff_t) == sizeof(long)) {
            MPI_TYPE_PTRDIFF = MPI_LONG;
        }
        if(sizeof(ptrdiff_t) == sizeof(int)) {
            MPI_TYPE_PTRDIFF = MPI_INT;
        }
    }

    MPI_Type_contiguous(d->rsize, MPI_BYTE, &o->MPI_TYPE_RADIX);
    MPI_Type_commit(&o->MPI_TYPE_RADIX);

}
static void _destroy_radix_sort_mpi(struct crmpistruct * o) {
    MPI_Type_free(&o->MPI_TYPE_RADIX);
}

static void _find_Pmax_Pmin_C(void * mybase, size_t mynmemb, 
        char * Pmax, char * Pmin, 
        ptrdiff_t * C,
        struct crstruct * d,
        struct crmpistruct * o);

int radix_sort_mpi(void * mybase, size_t mynmemb, size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg, 
        MPI_Comm comm) {
    struct crstruct d;
    struct crmpistruct o;

    struct piter pi;

    size_t nmemb;

    _setup_radix_sort(&d, size, radix, rsize, arg);
    _setup_radix_sort_mpi(&o, &d, comm);

    char Pmax[d.rsize];
    char Pmin[d.rsize];

    char P[d.rsize * (o.NTask - 1)];

    ptrdiff_t C[o.NTask + 1];  /* desired counts */

    ptrdiff_t myCLT[o.NTask + 1]; /* counts of less than P */
    ptrdiff_t CLT[o.NTask + 1]; 

    ptrdiff_t myCLE[o.NTask + 1]; /* counts of less than or equal to P */
    ptrdiff_t CLE[o.NTask + 1]; 

    int SendCount[o.NTask];
    int SendDispl[o.NTask];
    int RecvCount[o.NTask];
    int RecvDispl[o.NTask];

    int NTask1 = o.NTask + 1;
    ptrdiff_t GL_CLT[o.NTask * NTask1];
    ptrdiff_t GL_CLE[o.NTask * NTask1];
    ptrdiff_t GL_C[o.NTask * NTask1];



    MPI_Allreduce(&mynmemb, &nmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, o.comm);

    if(nmemb == 1) goto exec_empty_array;

    /* and sort the local array */
    radix_sort(mybase, mynmemb, d.size, d.radix, d.rsize, d.arg);

    _find_Pmax_Pmin_C(mybase, mynmemb, Pmax, Pmin, C, &d, &o);

    memset(P, 0, d.rsize * (o.NTask -1));

    piter_init(&pi, Pmin, Pmax, o.NTask - 1, &d);

    int iter = 0;
    int done = 0;

    while(!done) {
        iter ++;
        piter_bisect(&pi, P);

        _histogram(P, o.NTask - 1, mybase, mynmemb, myCLT, myCLE, &d);

        MPI_Allreduce(myCLT, CLT, o.NTask + 1, 
                MPI_TYPE_PTRDIFF, MPI_SUM, o.comm);
        MPI_Allreduce(myCLE, CLE, o.NTask + 1, 
                MPI_TYPE_PTRDIFF, MPI_SUM, o.comm);

        piter_accept(&pi, P, C, CLT, CLE);
#if 0
        {
            int k;
            for(k = 0; k < o.NTask; k ++) {
                MPI_Barrier(o.comm);
                int i;
                if(o.ThisTask != k) continue;
                
                printf("P (%d): PMin %d PMax %d P ", 
                        o.ThisTask, 
                        *(int*) Pmin,
                        *(int*) Pmax
                        );
                for(i = 0; i < o.NTask - 1; i ++) {
                    printf(" %d ", ((int*) P) [i]);
                }
                printf("\n");

                printf("C (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%d ", C[i]);
                }
                printf("\n");
                printf("CLT (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%d ", CLT[i]);
                }
                printf("\n");
                printf("CLE (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%d ", CLE[i]);
                }
                printf("\n");

            }
        }
#endif
        done = piter_all_done(&pi);
    }

    piter_destroy(&pi);

    _histogram(P, o.NTask - 1, mybase, mynmemb, myCLT, myCLE, &d);

    MPI_Allgather(myCLT, o.NTask + 1, MPI_TYPE_PTRDIFF, 
            GL_CLT, o.NTask + 1, MPI_TYPE_PTRDIFF, o.comm);
    MPI_Allgather(myCLE, o.NTask + 1, MPI_TYPE_PTRDIFF, 
            GL_CLE, o.NTask + 1, MPI_TYPE_PTRDIFF, o.comm);

    _solve_for_layout(o.NTask, C, GL_CLT, GL_CLE, GL_C);

    char * buffer = malloc(d.size * mynmemb);

    int i;
    for(i = 0; i < o.NTask; i ++) {
        SendCount[i] = GL_C[o.ThisTask * NTask1 + i + 1] - GL_C[o.ThisTask * NTask1 + i];
    }

    MPI_Alltoall(SendCount, 1, MPI_INT,
            RecvCount, 1, MPI_INT, o.comm);

    SendDispl[0] = 0;
    RecvDispl[0] = 0;
    for(i = 1; i < o.NTask; i ++) {
        SendDispl[i] = SendDispl[i - 1] + SendCount[i - 1];
        RecvDispl[i] = RecvDispl[i - 1] + RecvCount[i - 1];
        if(SendDispl[i] != GL_C[o.ThisTask * NTask1 + i]) {
            fprintf(stderr, "SendDispl error\n");
            abort();
        }
    }

#if 0
    {
        int k;
        for(k = 0; k < o.NTask; k ++) {
            MPI_Barrier(o.comm);

            if(o.ThisTask != k) continue;
            
            printf("P (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask - 1; i ++) {
                printf("%d ", ((int*) P) [i]);
            }
            printf("\n");

            printf("C (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", C[i]);
            }
            printf("\n");
            printf("CLT (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", CLT[i]);
            }
            printf("\n");
            printf("CLE (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", CLE[i]);
            }
            printf("\n");

            printf("MyC (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", GL_C[o.ThisTask * NTask1 + i]);
            }
            printf("\n");
            printf("MyCLT (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myCLT[i]);
            }
            printf("\n");

            printf("MyCLE (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myCLE[i]);
            }
            printf("\n");

            printf("Send Count(%d): ", o.ThisTask);
            for(i = 0; i < o.NTask; i ++) {
                printf("%d ", SendCount[i]);
            }
            printf("\n");
            printf("My data(%d): ", o.ThisTask);
            for(i = 0; i < mynmemb; i ++) {
                printf("%d ", ((int*) mybase)[i]);
            }
            printf("\n");
        }
    }
#endif
    MPI_Datatype MPI_TYPE_DATA;
    MPI_Type_contiguous(d.size, MPI_BYTE, &MPI_TYPE_DATA);
    MPI_Type_commit(&MPI_TYPE_DATA);
    MPI_Alltoallv(
            mybase, SendCount, SendDispl, MPI_TYPE_DATA,
            buffer, RecvCount, RecvDispl, MPI_TYPE_DATA, 
            o.comm);
    MPI_Type_free(&MPI_TYPE_DATA);

    memcpy(mybase, buffer, mynmemb * d.size);
    free(buffer);

    radix_sort(mybase, mynmemb, d.size, d.radix, d.rsize, d.arg);

exec_empty_array:
    _destroy_radix_sort_mpi(&o);
}

static void _find_Pmax_Pmin_C(void * mybase, size_t mynmemb, 
        char * Pmax, char * Pmin, 
        ptrdiff_t * C,
        struct crstruct * d,
        struct crmpistruct * o) {
    memset(Pmax, 0, d->rsize);
    memset(Pmin, -1, d->rsize);

    char myPmax[d->rsize];
    char myPmin[d->rsize];

    size_t eachnmemb[o->NTask];
    char eachPmax[d->rsize * o->NTask];
    char eachPmin[d->rsize * o->NTask];
    int i;

    if(mynmemb > 0) {
        d->radix(mybase + (mynmemb - 1) * d->size, myPmax, d->arg);
        d->radix(mybase, myPmin, d->arg);
    } else {
        memset(myPmin, 0, d->rsize);
        memset(myPmax, 0, d->rsize);
    }

    MPI_Allgather(&mynmemb, 1, MPI_TYPE_PTRDIFF, 
            eachnmemb, 1, MPI_TYPE_PTRDIFF, o->comm);
    MPI_Allgather(myPmax, 1, o->MPI_TYPE_RADIX, 
            eachPmax, 1, o->MPI_TYPE_RADIX, o->comm);
    MPI_Allgather(myPmin, 1, o->MPI_TYPE_RADIX, 
            eachPmin, 1, o->MPI_TYPE_RADIX, o->comm);


    C[0] = 0;
    for(i = 0; i < o->NTask; i ++) {
        C[i + 1] = C[i] + eachnmemb[i];
        if(eachnmemb[i] == 0) continue;

        if(d->compar(eachPmax + i * d->rsize, Pmax, d->rsize) > 0) {
            memcpy(Pmax, eachPmax + i * d->rsize, d->rsize);
        }
        if(d->compar(eachPmin + i * d->rsize, Pmin, d->rsize) < 0) {
            memcpy(Pmin, eachPmin + i * d->rsize, d->rsize);
        }
    }
}
