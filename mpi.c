#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

#define  N  (2*2*2*2*2*2+2)
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
    int rank, size;

    MPI_Init(&an, &as);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    init(A);

    double start = MPI_Wtime();

    for(it=1; it<=itmax; it++)
    {
        eps = 0.0;
        relax_diagonal(A);

        double global_eps;
        MPI_Allreduce(&eps, &global_eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        eps = global_eps;

        if (eps < maxeps) break;
    }

    if (rank == 0) {
        verify(A);
    }

    double end = MPI_Wtime();

    MPI_Reduce(&local_end, &global_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Elapsed time: %f\n", end - start);
    }

    MPI_Finalize();
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

    for (sum = 2; sum <= 2 * (N - 2); sum++) {

        #pragma omp parallel for reduction(max:eps) firstprivate(j)
        for (i = Max(1, sum - (N - 2)); i <= Min(sum - 1, N - 2); i++) {
            j = sum - i; 
            double e = A[i][j];
            A[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4.0;
            eps = Max(eps, fabs(e - A[i][j]));
        }
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank > 0) {
        MPI_Send(A[1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(A[0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank < size - 1) {
        MPI_Recv(A[N-1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(A[N-2], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
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
    printf("N = %d\n", N);
    printf("S = %f\n", s);
}
