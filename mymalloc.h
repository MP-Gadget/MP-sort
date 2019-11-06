/* This function contains stubs for the memory allocation routines used in MP-Gadget,
 * so we can use the same function names and thus easily copy the file across*/
#ifndef _MYMALLOC_H_
#define _MYMALLOC_H_

#define  mymalloc(name, size)            malloc(size)
#define  mymalloc2(name, size)           malloc(size)

#define  myfree(x)                 free(x)

#define  ta_malloc(name, type, nele)            (type*) malloc(sizeof(type) * (nele))
#define  ta_malloc2(name, type, nele)           (type*) malloc(sizeof(type) * (nele))
#define  ta_free(p) free(p)

#endif
