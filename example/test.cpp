#include <mpi.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, n = 5931, N = 8 * n;
    double t0, t1, tc, *a, *b, *c, *x, *y;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    a = malloc(3*n*sizeof(double));  // 连续内存分配
    b = a + n; c = b + n; x = malloc(n*sizeof(double));
    y = calloc(n, sizeof(double));

    if (rank == 0) {  // 主进程初始化数据
        double* full = malloc(4*N*sizeof(double));
        for(int i=0; i<3*N; i++) 
            full[i] = (i%3==0) ? 6+4*drand48() : 
                     (i%3==1) ? -2-drand48() : -5+2*drand48();
        for(int i=0; i<N; i++) 
            full[3*N+i] = 374 + 594*drand48();
        MPI_Scatter(full, 3*n, MPI_DOUBLE, a, 3*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(full+3*N, n, MPI_DOUBLE, x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(full);
    } else {
        MPI_Scatter(NULL, 3*n, MPI_DOUBLE, a, 3*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, n, MPI_DOUBLE, x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double bd[2] = {x[0], x[n-1]}, neighbor[2];
    MPI_Request req[4];
    t0 = MPI_Wtime();
    
    // 边界通信（非阻塞）
    MPI_Isend(&bd[0], 1, MPI_DOUBLE, (rank-1+8)%8, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&neighbor[1], 1, MPI_DOUBLE, (rank+1)%8, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(&bd[1], 1, MPI_DOUBLE, (rank+1)%8, 1, MPI_COMM_WORLD, &req[2]); 
    MPI_Irecv(&neighbor[0], 1, MPI_DOUBLE, (rank-1+8)%8, 1, MPI_COMM_WORLD, &req[3]);
    MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    tc = MPI_Wtime() - t0;

    // 核心计算
    y[0] = a[0]*x[0] + b[0]*x[1] + ((rank>0) ? c[0]*neighbor[0] : 0);
    for(int i=1; i<n-1; i++) 
        y[i] = a[i]*x[i] + b[i]*x[i+1] + c[i]*x[i-1];
    y[n-1] = a[n-1]*x[n-1] + c[n-1]*x[n-2] + ((rank<7) ? b[n-1]*neighbor[1] : 0);
    
    double *full_y = NULL, t_par = MPI_Wtime() - t0, t_ser, max_err;
    MPI_Gather(y, n, MPI_DOUBLE, (rank==0)?(full_y=malloc(N*sizeof(double))):NULL, 
              n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {  // 串行验证
        t0 = MPI_Wtime();
        double *sx = full_y + N, *sy = sx + N;  // 复用内存
        for(int i=0; i<N; i++) {
            sy[i] = a[i]*sx[i];
            if(i>0) sy[i] += c[i]*sx[i-1];
            if(i<N-1) sy[i] += b[i]*sx[i+1];
        }
        t_ser = MPI_Wtime() - t0;
        for(int i=max_err=0; i<N; i++) 
            if(fabs(full_y[i]-sy[i]) > max_err) max_err = fabs(full_y[i]-sy[i]);
        printf("Max error: %.2e\nSpeedup: %.2f\nComm time: %.4fs\n", 
              max_err, t_ser/t_par, tc);
        free(full_y);
    }
    
    free(a); free(x); free(y);
    MPI_Finalize();
    return 0;
}

