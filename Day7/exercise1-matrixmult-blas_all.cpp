#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include "mkl.h"

int mnk=4;

double mytime(){
  timeval v;
  gettimeofday(&v,0);
  return v.tv_sec+v.tv_usec/1000000.0;
}

void matrixmul_mnk(double* c,double* a,double* b){
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
	      mnk, mnk, mnk, 1, a, mnk, b, mnk, 1, c, mnk);
}


void matrixmul_mnk_c(double* c,double* a, double* b){
		for(int i=0; i<4; i+=1){
			for(int j=0; j<4; j+=1){
			#pragma unroll(4)
   				for(int k=0; k<4; k+=1){
					c[(i*4)+j]=c[(i*4)+j]+a[(i*4)+k]*b[(k*4)+j];
				}
		}
	}
}


void matrixmul_mnk_an(double* c,double* a, double* b){
		for(int i=0; i<4; i+=1){
			#pragma unroll(4)
   			for(int k=0; k<4; k+=1){
  				 c[(i*4):4]=c[(i*4):4]+a[(i*4)+k]*b[(k*4):4];
			}
		}
}


void matrixmul_mkl_intrinsic(double* c, double* a, double* b){
    __m256d a_vec, b_vec, c_vec; // vectors with 4 double entries
    for(int i=0;i<4*4;i+=4){
  		c_vec = _mm256_load_pd(&c[i]); //Bring the line of vector c
        	    for (int j = 0; j < 4; j++) {
					b_vec = _mm256_load_pd(&b[j*4]); //Load the row of B
					a_vec = _mm256_set1_pd(a[i+j]); //Set the correspondence value 
					c_vec = _mm256_add_pd(_mm256_mul_pd(a_vec,b_vec), c_vec); 
				}
		_mm256_store_pd(&c[i], c_vec);
		                              }
}

int main(void){
  int iter=10;
  int nmatrices=100000;
  int size=mnk*mnk*nmatrices;
  printf("size= %d\n" , size);
  double* a= (double*) _mm_malloc(sizeof(double)*size,64);
  double* b= (double*) _mm_malloc(sizeof(double)*size,64);
  double* c= (double*) _mm_malloc(sizeof(double)*size,64);
  double time1,time2;
  for(int i=0;i<size;i++){
    a[i]=rand();
    b[i]=rand();
    c[i]=rand();
  }

//--------------------------------LAPACK-----------------------------------
  time1=mytime();
  for(int n=0;n<iter;n++){
    for(int i=0;i<size;i+=mnk*mnk){
         matrixmul_mnk(&c[i],&a[i],&b[i]);
		 }
  }
  time2=mytime();

  //-------------------------------C CODE-----------------------------------
 
  printf("time Lapack= %f s\n", time2-time1);
  printf("perf Lapack= %f GFLOPs\n", (2.0*mnk*mnk*mnk*nmatrices*iter)/(time2-time1)/1000.0/1000.0/1000.0);

  time1=mytime();
  for(int n=0;n<iter;n++){ 
	     for(int j=0; j<size; j+=mnk*mnk ){
    	 #pragma forceinline 
		 matrixmul_mnk_c(&c[j],&a[j],&b[j]);   	
 		 }
   }	  
  time2=mytime();

  printf("\ntime c code  = %f s\n", time2-time1);
  printf("perf c code= %f GFLOPs\n", (2.0*mnk*mnk*mnk*nmatrices*iter)/(time2-time1)/1000.0/1000.0/1000.0);


//--------------------------------ARRAY NOTATION-----------------------------

  time1=mytime();
  for(int n=0;n<iter;n++){ 
	     for(int j=0; j<size; j+=mnk*mnk ){
    	 #pragma forceinline 
		 matrixmul_mnk_an(&c[j],&a[j],&b[j]);   	
//		for(int i=0; i<4; i+=1){
//   			for(int k=0; k<4; k+=1){
//  			 c[(i*4):4]=c[(i*4):4]+a[(i*4)+k]*b[(k*4):4];
 		 }
   }	  
  time2=mytime();

  printf("\ntime Array notation  = %f s\n", time2-time1);
  printf("perf Array notation= %f GFLOPs\n", (2.0*mnk*mnk*mnk*nmatrices*iter)/(time2-time1)/1000.0/1000.0/1000.0);
  
  
  //---------------------------------INTRINSIC NOTATION------------------------
  time1=mytime();
  for(int n=0;n<iter;n++){ 
	     for(int j=0; j<size; j+=mnk*mnk ){
		 matrixmul_mkl_intrinsic(&c[j],&a[j],&b[j]);   	
 		 }
   }	  
  time2=mytime();

  printf("\ntime Intrinsic  = %f s\n", time2-time1);
  printf("perf Intrinsic= %f GFLOPs\n", (2.0*mnk*mnk*mnk*nmatrices*iter)/(time2-time1)/1000.0/1000.0/1000.0);
}
