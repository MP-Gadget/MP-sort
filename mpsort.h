void radix_sort(void * base, size_t nmemb, size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg),
        size_t rsize,
        void * arg);

#ifdef _OPENMP
/* openmp support */
void mpsort_omp(void * base, size_t nmemb, size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg);

#endif

#ifdef MPI_VERSION
/* MPI support */
#define MPSORT_DISABLE_SPARSE_ALLTOALLV (1 << 1)
#define MPSORT_DISABLE_IALLREDUCE (1 << 2)
#define MPSORT_DISABLE_GATHER_SORT (1 << 3)
#define MPSORT_REQUIRE_GATHER_SORT (1 << 4)
#define MPSORT_REQUIRE_SPARSE_ALLTOALLV (1 << 6)

void mpsort_mpi_set_options(int options);
int mpsort_mpi_has_options(int options);
void mpsort_mpi_unset_options(int options);

void mpsort_mpi(void * base, size_t nmemb, size_t elsize,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg, MPI_Comm comm);
void mpsort_mpi_newarray(void * base, size_t nmemb, 
        void * out, size_t outnmemb,
        size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg), 
        size_t rsize, 
        void * arg, MPI_Comm comm);

void mpsort_mpi_report_last_run();
#endif
