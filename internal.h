#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


struct crstruct {
    size_t size;
    size_t rsize;
    void * arg;
    void (*radix)(const void * ptr, void * radix, void * arg);
    int (*compar)(const void * r1, const void * r2, size_t rsize);
    void (*bisect)(void * r, const void * r1, const void * r2, size_t rsize);
};

int _compute_and_compar_radix(const void * p1, const void * p2, void * arg);
void _setup_radix_sort(
        struct crstruct *d,
        size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg);

