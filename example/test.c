#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 5931
#define TOTAL_SIZE (8*N)

int main(int argc, char *argv[]) {
    int myid, numprocs, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    double *a, *b, *c, *x, *y;
    double t_start, t_end, comm_time, max_err;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stdout, "Process %d of %d on %s\n", myid, numprocs, processor_name);
    fflush(stdout);

    // 内存分配
    a = malloc(3*N*sizeof(double));
    b = a + N;
    c = b + N;
    x = malloc(N*sizeof(double));
    y = calloc(N, sizeof(double));

    // 主进程初始化数据
    if (myid == 0) {
        double *full_data = malloc(4*TOTAL_SIZE*sizeof(double));
        srand(time(NULL));
        for(int i=0; i<3*TOTAL_SIZE; i++)
            full_data[i] = (i%3 == 0) ? 6.0 + 4.0*rand()/RAND_MAX :
                          (i%3 == 1) ? -2.0 - 1.0*rand()/RAND_MAX :
                                      -5.0 + 2.0*rand()/RAND_MAX;
        for(int i=0; i<TOTAL_SIZE; i++)
            full_data[3*TOTAL_SIZE+i] = 374.0 + 594.0*rand()/RAND_MAX;
        
        MPI_Scatter(full_data, 3*N, MPI_DOUBLE, a, 3*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(full_data+3*TOTAL_SIZE, N, MPI_DOUBLE, x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(full_data);
        t_start = MPI_Wtime();
    } else {
        MPI_Scatter(NULL, 3*N, MPI_DOUBLE, a, 3*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, N, MPI_DOUBLE, x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // 边界通信
    double send_buf[2] = {x[0], x[N-1]}, recv_buf[2];
    MPI_Request req[4];
    
    double comm_start = MPI_Wtime();
    MPI_Isend(send_buf, 1, MPI_DOUBLE, (myid-1+8)%8, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(recv_buf+1, 1, MPI_DOUBLE, (myid+1)%8, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(send_buf+1, 1, MPI_DOUBLE, (myid+1)%8, 1, MPI_COMM_WORLD, &req[2]);
    MPI_Irecv(recv_buf, 1, MPI_DOUBLE, (myid-1+8)%8, 1, MPI_COMM_WORLD, &req[3]);
    MPI_Waitall(4, req, MPI_STATUS_IGNORE);
    comm_time = MPI_Wtime() - comm_start;

    // 核心计算
    y[0] = a[0]*x[0] + b[0]*x[1] + (myid>0 ? c[0]*recv_buf[0] : 0);
    for(int i=1; i<N-1; i++)
        y[i] = a[i]*x[i] + b[i]*x[i+1] + c[i]*x[i-1];
    y[N-1] = a[N-1]*x[N-1] + c[N-1]*x[N-2] + (myid<7 ? b[N-1]*recv_buf[1] : 0);

    // 结果收集与验证
    double *final_result = NULL;
    if(myid == 0) final_result = malloc(TOTAL_SIZE*sizeof(double));
    
    MPI_Gather(y, N, MPI_DOUBLE, final_result, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(myid == 0) {
        t_end = MPI_Wtime();
        
        // 串行验证
        double *sx = final_result + TOTAL_SIZE;
        double *sy = sx + TOTAL_SIZE;
        double serial_time = MPI_Wtime();
        for(int i=0; i<TOTAL_SIZE; i++) {
            sy[i] = a[i]*sx[i];
            if(i>0) sy[i] += c[i]*sx[i-1];
            if(i<TOTAL_SIZE-1) sy[i] += b[i]*sx[i+1];
        }
        serial_time = MPI_Wtime() - serial_time;

        // 计算最大误差
        max_err = 0;
        for(int i=0; i<TOTAL_SIZE; i++)
            if(fabs(final_result[i]-sy[i]) > max_err)
                max_err = fabs(final_result[i]-sy[i]);

        printf("\nMax error: %.2e\nParallel time: %.6fs\nSerial time: %.6fs\n"
               "Speedup ratio: %.2f\nComm time: %.6fs\n",
               max_err, t_end-t_start, serial_time, 
               serial_time/(t_end-t_start), comm_time);
        
        free(final_result);
    }

    free(a); free(x); free(y);
    MPI_Finalize();
    return 0;
}

