#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

#ifndef N
#define  N  (2*2*2*2*2*2+2)
#endif

double   maxeps = 0.1e-7;
int itmax = 100;
double eps;

double A [N][N];

void relax();
void init();
void verify(); 
void relax_diagonal();

int main(int an, char **as)
{	
	int it;

	init(A);

	double start = omp_get_wtime();
	
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax_diagonal(A);
		if (eps < maxeps) break;
	}
	
	verify(A);

	double end = omp_get_wtime();

	printf("Elapsed time: %f\nUsed threads: %d\n", end - start, omp_get_max_threads());
	return 0;
}


void init(double A [N][N])
{ 
	int i, j;

	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1){
		A[i][j]= 0.;}
		else A[i][j]= ( 1. + i + j ) ;
	}
} 


void relax_diagonal(double A[N][N]) {
    int i, j, sum;
    #pragma omp parallel
    {
    #pragma omp single
    {

    for (sum = 2; sum <= 2 * (N - 2); sum++) {

		#pragma omp taskloop firstprivate(j) shared(A, eps) 
        for (i = Max(1, sum - (N - 2)); i <= Min(sum - 1, N - 2); i++) {
            j = sum - i; 
			double e = A[i][j];
			A[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4.0;
			eps = Max(eps, fabs(e - A[i][j]));
        }
        }
    }
    }
}

void relax(double A [N][N])
{
	int i, j;

	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	{ 
		double e;
		e=A[i][j];
		A[i][j]=(A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
		eps=Max(eps, fabs(e-A[i][j]));
	}    
}


void verify(double A [N][N])
{ 
	int i, j;
	double s;

	s=0.;
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("  S = %f\n",s);
}