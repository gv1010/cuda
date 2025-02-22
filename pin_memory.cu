#include <iostream>
#include <cuda_runtime.h>


int main(){
    int N = 1048576;
    int bytes = N * sizeof(int);
    int *h_a = (int *)malloc(bytes);
    int *h_aPinned;
    int *d_a;

    cudaMallocHost((void**)&h_aPinned, bytes) ;
    cudaMalloc((void**)&d_a, bytes);

    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf(" Pageable Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    cudaEvent_t pstartEvent, pstopEvent; 
    cudaEventCreate(&pstartEvent);
    cudaEventCreate(&pstopEvent);
    cudaEventRecord(pstartEvent, 0);
    cudaMemcpy(d_a, h_aPinned, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(pstopEvent, 0);
    cudaEventSynchronize(pstopEvent);

    float ptime;
    cudaEventElapsedTime(&ptime, pstartEvent, pstopEvent);
    printf(" Pinned Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / ptime);
    
    cudaFreeHost(h_aPinned);
    cudaFree(d_a);
    free(h_a);
    return 0;
}
