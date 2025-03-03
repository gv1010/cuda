%%writefile shared_mat_mul.cu

#include <iostream>
#include <cuda_runtime.h>

# define TILE_WIDTH 32

__global__ void navie_mat_mul(float *P_A, float *P_B,float *P_C, int Width){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < Width && col < Width){
        for (int k=0; k<Width; k++){
            P_C[row*Width+col] += P_A[row*Width+k] * P_B[k*Width+col];
        }
    }
}

__global__ void  mat_mul(float *A, float *B, float *C, int width){
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float element_value = 0;

    for (int t=0; t < int (width+TILE_WIDTH - 1)/TILE_WIDTH; t ++){

        if ((row < width) && ((t*TILE_WIDTH + tx) < width)){
            As[ty][tx] = A[row*width + t*TILE_WIDTH + tx];
        }
        else{
            As[ty][tx] = 0.0f;
        }
    
        if ((col < width) && ((t*TILE_WIDTH + ty) < width)){
            Bs[ty][tx] = B[(t*TILE_WIDTH + ty)*width + col];
        }
        else{
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        
        for (int k=0; k < TILE_WIDTH; k++){
            element_value += As[ty][k]* Bs[k][tx];
        }
        __syncthreads();

    }
    if ((row < width) && (col < width)){
        C[row*width + col] = element_value;
    }

}


int main(){
    int rows = 1024;
    int cols = 1024;
    int size = rows * cols; 
    float *hA, *hB, *hC, *hC_naive; // Host arrays
    float *dA, *dB, *dC, *dC_naive; // Device arrays

    // Allocate memory on host
    hA = new float[size];
    hB = new float[size];
    hC = new float[size];
    hC_naive = new float[size];

    // Allocate memory on device
    cudaMalloc((void **)&dA, size * sizeof(float));
    cudaMalloc((void **)&dB, size * sizeof(float));
    cudaMalloc((void **)&dC, size * sizeof(float));

    cudaMalloc((void **)&dC_naive, size * sizeof(float));

    // Initialize host arrays (example values)
    for (int row = 0; row < rows; row ++){
        for (int col = 0; col < cols; col++){
            // hA[row*cols + col] = row*cols + col;
            hA[row*cols + col] = 1;
        }
    }

    // Initialize host arrays (example values)
    for (int row = 0; row < rows; row ++){
        for (int col = 0; col < cols; col++){
            // hB[row*cols + col] = row*cols + col;
            hB[row*cols + col] = 1;
        }
    }

    // Copy data from host to device
    cudaMemcpy(dA, hA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size * sizeof(float), cudaMemcpyHostToDevice);


    // Kernel launch configuration
    dim3 blockSize(32, 32, 1);
    //dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1);

    printf("%25s%25s%25s\n", "Exectution", "Bandwidth (GB/s)", "GPU Time (ms)");
    printf("\n");

    printf("%25s","Matrix Mul");

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    float total_milliseconds1 = 0;
    int num_runs = 100;

    for (int i=0; i < num_runs; i++){
        cudaEventRecord(start1);
        mat_mul<<<1, blockSize>>>(dA, dB, dC, rows);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start1, stop1);
        total_milliseconds1 += milliseconds1;
    }
    float average_milliseconds1 = total_milliseconds1 / num_runs;
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaMemcpy(hC, dC, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f%25.4f\n", 2 * size * sizeof(float)* 1e-6 / average_milliseconds1, average_milliseconds1 );


    
    printf("%25s","Navie Matrix Mul");

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float total_milliseconds2 = 0;
    int num_runs1 = 100;

    for (int i=0; i < num_runs1; i++){
        cudaEventRecord(start2);
        navie_mat_mul<<<1, blockSize>>>(dA, dB, dC_naive, rows);
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        float milliseconds2 = 0;
        cudaEventElapsedTime(&milliseconds2, start2, stop2);
        total_milliseconds2 += milliseconds2;
    }
    float average_milliseconds2 = total_milliseconds2 / num_runs1;
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaMemcpy(hC_naive, dC_naive, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%20.4f%25.4f\n", 2 * size * sizeof(float)* 1e-6 / average_milliseconds2, average_milliseconds2 );

    
    printf("hA: \n ");
    for (int row = 0; row < 5; row ++){
        for (int col = 0; col < 5; col++){
            printf("%0.0f ", hA[row*cols + col]);
        }
        printf("\n");
    }
    
    printf("hB: \n");
    for (int row = 0; row < 5; row ++){
        for (int col = 0; col < 5; col++){
            printf("%0.0f ", hB[row*cols + col]);
        }
        printf("\n");
    }

    printf("\n\n");
    printf("hC: \n");
    for (int row = 0; row < 5; row ++){
        for (int col = 0; col < 5; col++){
            printf("%0.f ", hC[row*cols + col]);
        }
        printf("\n");
    }

    printf("\n\n");
    printf("hC_naive: \n");
    for (int row = 0; row < 5; row ++){
        for (int col = 0; col < 5; col++){
            printf("%0.f ", hC_naive[row*cols + col]/num_runs1);
        }
        printf("\n");
    }
    return 0;

}
