#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <xmmintrin.h>
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

int main(void){
  int iter=10;
  int nmatrices=1000;
  int size=mnk*mnk*nmatrices;
  printf("size= %d\n" , size);
  double* a= (double*) _mm_malloc(sizeof(double)*size,64);
  double* b= (double*) _mm_malloc(sizeof(double)*size,64);
  double* c= (double*) _mm_malloc(sizeof(double)*size,64);
  double time1,time2;
  for(int i=0;i<4;i++){
    a[i]=rand();
    b[i]=rand();
    c[i]=rand();
  }
  
  time1=mytime();
  for(int n=0;n<iter;n++){
   // for(int i=0;i<size;i+=mnk*mnk){
   //if(mnk==4){
   for(int j=0; j<nmatrices; j++){
   		for(int i=0; i<4; i+=1){
   			for(int k=0; k<4; k+=1){
  			 c[(i*4):4]=c[(i*4):4]+a[(i*4)+k]*b[(k*4):4];
  			 }
  		 }
	}
 // }// else{
	// matrixmul_mnk(&c[0:size],&a[0:size],&b[0:size]);
//	} //you code goes here	
      //matrixmul_mnk_opt1(&c[i],&a[i],&b[i]);
   // }
  }
  time2=mytime();



  printf("time = %f s\n", time2-time1);
  printf("perf = %f GFLOPs\n", (2.0*mnk*mnk*mnk*nmatrices*iter)/(time2-time1)/1000.0/1000.0/1000.0);
}
