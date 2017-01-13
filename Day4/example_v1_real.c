// This is a simple standalone example. See README.txt

#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"     // if you need CUBLAS v2, include before magma.h
#include "magma.h"
#include "magma_lapack.h"  // if you need BLAS & LAPACK

#include <sys/time.h>

double cWtime(void)
{
    struct timeval tp;
	    gettimeofday( &tp, NULL );
		    return tp.tv_sec + 1e-6 * tp.tv_usec;
}


// ------------------------------------------------------------
// Replace with your code to initialize the A matrix.
// This simply initializes it to random values.
// Note that A is stored column-wise, not row-wise.
//
// m   - number of rows,    m >= 0.
// n   - number of columns, n >= 0.
// A   - m-by-n array of size lda*n.
// lda - leading dimension of A, lda >= m.
//
// When lda > m, rows (m, ..., lda-1) below the bottom of the matrix are ignored.
// This is helpful for working with sub-matrices, and for aligning the top
// of columns to memory boundaries (or avoiding such alignment).
// Significantly better memory performance is achieved by having the outer loop
// over columns (j), and the inner loop over rows (i), than the reverse.
void dfill_matrix(
    magma_int_t m, magma_int_t n, double *A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for (j=0; j < n; ++j) {
        for (i=0; i < m; ++i) {
            A(i,j) = MAGMA_D_MAKE( rand() / ((double) RAND_MAX), rand()/((double) RAND_MAX) );  
        }
    }
    
    #undef A
}


// ------------------------------------------------------------
// Replace with your code to initialize the X rhs.
void dfill_rhs(
    magma_int_t m, magma_int_t nrhs, double *X, magma_int_t ldx )
{
    dfill_matrix( m, nrhs, X, ldx );
}


// ------------------------------------------------------------
// Replace with your code to initialize the dA matrix on the GPU device.
// This simply leverages the CPU version above to initialize it to random values,
// and copies the matrix to the GPU.
void dfill_matrix_gpu(
    magma_int_t m, magma_int_t n, double *dA, magma_int_t ldda )
{
    double *A;
    int lda = ldda;
    magma_dmalloc_cpu( &A, m*lda );
    if (A == NULL) {
        fprintf( stderr, "malloc failed\n" );
        return;
    }
    dfill_matrix( m, n, A, lda );
    magma_dsetmatrix( m, n, A, lda, dA, ldda );
    magma_free_cpu( A );
}


// ------------------------------------------------------------
// Replace with your code to initialize the dX rhs on the GPU device.
void dfill_rhs_gpu(
    magma_int_t m, magma_int_t nrhs, double *dX, magma_int_t lddx )
{
    dfill_matrix_gpu( m, nrhs, dX, lddx );
}


// ------------------------------------------------------------
// Solve A * X = B, where A and X are stored in CPU host memory.
// Internally, MAGMA transfers data to the GPU device
// and uses a hybrid CPU + GPU algorithm.
void cpu_interface( magma_int_t n, magma_int_t nrhs )
{
    double *A=NULL, *X=NULL;
    magma_int_t *ipiv=NULL;
    magma_int_t lda  = n;
    magma_int_t ldx  = lda;
    magma_int_t info = 0;
    
    // magma_*malloc_cpu routines for CPU memory are type-safe and align to memory boundaries,
    // but you can use malloc or new if you prefer.
    magma_dmalloc_cpu( &A, lda*n );
    magma_dmalloc_cpu( &X, ldx*nrhs );
    magma_imalloc_cpu( &ipiv, n );
    if (A == NULL || X == NULL || ipiv == NULL) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    // Replace these with your code to initialize A and X
    dfill_matrix( n, n, A, lda );
    dfill_rhs( n, nrhs, X, ldx );
	double start =cWtime();
    
    magma_dgesv( n, 1, A, lda, ipiv, X, lda, &info );
	double stop =cWtime();
	double t=stop-start;
	printf(" TIME CPU %f\n", t  );
	printf("%d %g %g\n" ,n , t , 2.0 /3.0 * n*n*n/t);

    if (info != 0) {
        fprintf( stderr, "magma_zgesv failed with info=%d\n", info );
    }
    
    // TODO: use result in X
    
cleanup:
    magma_free_cpu( A );
    magma_free_cpu( X );
    magma_free_cpu( ipiv );
}


// ------------------------------------------------------------
// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
void gpu_interface( magma_int_t n, magma_int_t nrhs )
{
    double *dA=NULL, *dX=NULL;
    magma_int_t *ipiv=NULL;
    magma_int_t ldda = magma_roundup( n, 32 );  // round up to multiple of 32 for best GPU performance
    magma_int_t lddx = ldda;
    magma_int_t info = 0;
    
    // magma_*malloc routines for GPU memory are type-safe,
    // but you can use cudaMalloc if you prefer.
    magma_dmalloc( &dA, ldda*n );
    magma_dmalloc( &dX, lddx*nrhs );
    magma_imalloc_cpu( &ipiv, n );  // ipiv always on CPU
    if (dA == NULL || dX == NULL || ipiv == NULL) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    // Replace these with your code to initialize A and X
    dfill_matrix_gpu( n, n, dA, ldda );
    dfill_rhs_gpu( n, nrhs, dX, lddx );
	double start_gpu =cWtime();
    
    magma_dgesv_gpu( n, 1, dA, ldda, ipiv, dX, ldda, &info );
	double stop_gpu =cWtime();
	double t=stop_gpu-start_gpu;
	printf(" TIME GPU %f\n", t  );
	printf("%d %g %g\n" ,n , t , 2.0 /3.0 * n*n*n/t);
	

    if (info != 0) {
        fprintf( stderr, "magma_zgesv_gpu failed with info=%d\n", info );
    }
    
    // TODO: use result in dX
    
cleanup:
    magma_free( dA );
    magma_free( dX );
    magma_free_cpu( ipiv );
}


// ------------------------------------------------------------
int main( int argc, char** argv )
{
    magma_init();
    
    magma_int_t n = 3000;
    magma_int_t nrhs = 1;
    
    printf( "using MAGMA CPU interface\n" );

    cpu_interface( n, nrhs );
    printf( "using MAGMA GPU interface\n" );
    gpu_interface( n, nrhs );
	
//	printf("%d %g %g\n" ,n , t , 2.0 /3.0 * n*n*n/t);
    
	magma_finalize();
    return 0;
}
