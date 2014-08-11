
/* 
 * returns index of the last item satisfying 
 * [item] < P,
 *
 * returns -1 if [all] < P
 * */

static ptrdiff_t _bsearch_last_lt(void * P, 
    void * base, size_t nmemb, 
    struct crstruct * d) {

    if (nmemb == 0) return -1;

    char tmpradix[d->rsize];
    ptrdiff_t left = 0;
    ptrdiff_t right = nmemb - 1;

    d->radix((char*) base, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) >= 0) {
        return - 1;
    }
    d->radix((char*) base + right * d->size, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) < 0) {
        return nmemb - 1;
    }

    /* left <= i <= right*/
    /* [left] < P <= [right] */
    while(right > left + 1) {
        ptrdiff_t mid = ((right - left + 1) >> 1) + left;
        d->radix((char*) base + mid * d->size, tmpradix, d->arg);
        /* if [mid] < P , move left to mid */
        /* if [mid] >= P , move right to mid */
        int c1 = d->compar(tmpradix, P, d->rsize);
        if(c1 < 0) {
            left = mid;
        } else {
            right = mid;
        }
    } 
    return left;
}

/* 
 * returns index of the last item satisfying 
 * [item] <= P,
 *
 * */
static ptrdiff_t _bsearch_last_le(void * P, 
    void * base, size_t nmemb, 
    struct crstruct * d) {

    if (nmemb == 0) return -1;

    char tmpradix[d->rsize];
    ptrdiff_t left = 0;
    ptrdiff_t right = nmemb - 1;

    d->radix((char*) base, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) > 0) {
        return -1;
    }
    d->radix((char*) base + right * d->size, tmpradix, d->arg);
    if(d->compar(tmpradix, P, d->rsize) <= 0) {
        return nmemb - 1;
    }

    /* left <= i <= right*/
    /* [left] <= P < [right] */
    while(right > left + 1) {
        ptrdiff_t mid = ((right - left + 1) >> 1) + left;
        d->radix((char*) base + mid * d->size, tmpradix, d->arg);
        /* if [mid] <= P , move left to mid */
        /* if [mid] > P , move right to mid*/
        int c1 = d->compar(tmpradix, P, d->rsize);
        if(c1 <= 0) {
            left = mid;
        } else {
            right = mid;
        }
    } 
    return left;
}

/* 
 * do a histogram of mybase, based on bins defined in P.
 * P is an array of radix of length Plength,
 * myCLT, myCLE are of length Plength + 2
 *
 * myCLT[i + 1] is the count of items less than P[i]
 * myCLE[i + 1] is the count of items less than or equal to P[i]
 *
 * myCLT[0] is always 0
 * myCLT[Plength + 1] is always mynmemb
 *
 * */
static void _histogram(char * P, int Plength, void * mybase, size_t mynmemb, 
        ptrdiff_t * myCLT, ptrdiff_t * myCLE,
        struct crstruct * d) {
    int it;

    myCLT[0] = 0;
    myCLE[0] = 0;
    for(it = 0; it < Plength; it ++) {
        myCLT[it + 1] = _bsearch_last_lt(P + it * d->rsize, mybase, mynmemb, d) + 1;
        myCLE[it + 1] = _bsearch_last_le(P + it * d->rsize, mybase, mynmemb, d) + 1;
    }
    myCLT[it + 1] = mynmemb;
    myCLE[it + 1] = mynmemb;
}
#if 0
static ptrdiff_t _radix_count_lt_stupid(void * P, 
        void * base, size_t nmemb, 
        struct crstruct * d) {
    char tmpradix[d->rsize];
    ptrdiff_t i = 0;
    ptrdiff_t count = 0;
    for(i = 0; i < nmemb; i ++) {
        d->radix((char*) base + i * d->size, tmpradix, d->arg);
        if(d->compar(P, tmpradix, d->rsize) > 0) {
            count ++;
        } 
    }
    return count;
}
#endif
