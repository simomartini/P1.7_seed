#include <stdio.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include <time.h>

double mytime(){
  timeval v;
  gettimeofday(&v,0);
  return v.tv_sec+v.tv_usec/1000000.0;
}


int main(void){
  int size=100002;
  int iter=100002;
  double* f1= (double*) _mm_malloc(sizeof(double)*size,32);
  double* f2= (double*) _mm_malloc(sizeof(double)*size,32);
  double* tmpptr;
  double a,b;
  double c;
  double dx,dt;
  double time1,time2;

  // set some meaningfull parameters
  c=0.1;
  dx=0.01;
  dt=0.01;

  // precompute some values
  a=c*dt/dx;
  a=a*a;
  b=2*(1-a);

  // initialize to zero
  for(int i=0;i<size;i++){
    f1[i]=0;
    f2[i]=0;
  }
  // make some delta peaks
  f1[size/2]=0.1;
  f2[size/2]=-0.1;
  
  time1=mytime();
  for(int t=0;t<iter;t++){
    for(int i=1;i<size-1;i++){
      f2[i]=a*(f1[i+1]+f1[i-1])+b*f1[i]-f2[i];
    }
    tmpptr=f1;
    f1=f2;
    f2=tmpptr;
  }
  time2=mytime();
  
  // int case we need to look at the result
#ifdef _PRINT_
  for(int i=0;i<size;i++){
    printf("%f %f\n", dx*i,f1[i]);
  }
#else
  printf("time = %f\n", time2-time1);
#endif
}
