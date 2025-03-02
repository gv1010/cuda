#include <iostream>
#include <cuda_runtime.h>

const int TILE_DIM = 32;
const int num_runs = 100;

// Simple kernel to perform vector addition
__global__ void matrix_copy(float* in_matrix, float* out_matrix, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int block_rows = blockDim.y;
    int size = blockDim.x * gridDim.x; // as rows == cols
    if (x < n && y < n)
        for (int dim_y = 0; dim_y < gridDim.y; dim_y += block_rows){
            out_matrix[(dim_y+ y)*size + x] = in_matrix[(dim_y+ y)*size + x];
        }
}

__global__ void navie_transpose(float* in_matrix, float* out_matrix, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int block_rows = blockDim.y;
    int size = blockDim.x * gridDim.x; // as rows == cols
    if (x < n && y < n)
        for (int dim_y = 0; dim_y < gridDim.y; dim_y += block_rows){
            out_matrix[x*size + (dim_y + y)] = in_matrix[(dim_y + y)*size + x];
        }

}

__global__ void shared_copy(float* in_matrix, float* out_matrix, int n) {
    __shared__ float share[TILE_DIM * TILE_DIM];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int block_rows = blockDim.y;
    // int size = blockDim.x * gridDim.x; // as rows == cols
    if (x < n && y < n)
        share[(threadIdx.y)*blockDim.y + threadIdx.x] = in_matrix[(y)*n + x];
        __syncthreads();
        // for (int dim_y = 0; dim_y < gridDim.y; dim_y += block_rows)
        {
        out_matrix[(y)*n + x] = share[(threadIdx.x)*blockDim.x + threadIdx.y];
        }

    }

__global__ void shared_coalesced_bank_conflict(float *in_matrix, float *out_matrix, int n){
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int width = n;
    if (x < n && y < n){
        tile[threadIdx.y][threadIdx.x] = in_matrix[(y * width ) + threadIdx.x];
    }
    __syncthreads();
    
    x = blockIdx.y*blockDim.x + threadIdx.x;
    y = blockIdx.x*blockDim.y + threadIdx.y;
    if (x < n && y < n){
        out_matrix[(y*width) + x] = tile[threadIdx.x][threadIdx.y];
    }
 }

 __global__ void shared_coalesced_no_bank_conflict(float *in_matrix, float *out_matrix, int n){
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int width = n;
    if (x < n && y < n){
        tile[threadIdx.y][threadIdx.x] = in_matrix[(y * width ) + threadIdx.x];
    }
    __syncthreads();
    
    x = blockIdx.y*blockDim.x + threadIdx.x;
    y = blockIdx.x*blockDim.y + threadIdx.y;
    if (x < n && y < n){
        out_matrix[(y*width) + x] = tile[threadIdx.x][threadIdx.y];
    }
 }

int main() {
    int rows = 1024;
    int cols = 1024;
    int size = rows * cols; 
    float *c_data, *c_output; // Host arrays
    float *d_data, *d_output; // Device arrays

    // Allocate memory on host
    c_data = new float[size];
    c_output = new float[size];

    // Allocate memory on device
    cudaMalloc((void **)&d_data, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));
    // cudaMallocHost creates a pinned memory, avoids the overhead of pageable -> pinned memory
    // When executed cudaMallocHost, the execution time impacted, not all the time the pinned meory is efficient

    // Initialize host arrays (example values)
    for (int row = 0; row < rows; row ++){
        for (int col = 0; col < cols; col++){
            c_data[row*cols + col] = row*cols + col;
        }
    }

    // Copy data from host to device
    cudaMemcpy(d_data, c_data, size * sizeof(float), cudaMemcpyHostToDevice);


    // Kernel launch configuration
    dim3 blockSize(32, 32, 1);
    dim3 gridSize(32, 32, 1);

    printf("%25s%25s%25s\n", "Exectution", "Bandwidth (GB/s)", "GPU Time (ms)");
    printf("\n");

    printf("%25s","Matrix Copy");

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    float total_milliseconds1 = 0;
    for (int i=0; i < num_runs; i++){
        cudaEventRecord(start1);
        matrix_copy<<<gridSize, blockSize>>>(d_data, d_output, rows);
        //navie_transpose<<<gridSize, blockSize>>>(d_data, d_output, rows);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start1, stop1);
        total_milliseconds1 += milliseconds1;
    }
    float average_milliseconds1 = total_milliseconds1 / num_runs;
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaMemcpy(c_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f%25.4f\n", 2 * size * sizeof(float)* 1e-6 / average_milliseconds1, average_milliseconds1 );


    /*  -----------------------  */

    printf("%25s","Shared Copy");

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    
    float total_milliseconds2 = 0;
    for (int i=0; i < num_runs; i++){
        cudaEventRecord(start2);
        shared_copy<<<gridSize, blockSize>>>(d_data, d_output, rows);
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        float milliseconds2 = 0;
        cudaEventElapsedTime(&milliseconds2, start2, stop2);
        total_milliseconds2 += milliseconds2;
    }
    float average_milliseconds2 = total_milliseconds2 / num_runs;
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaMemcpy(c_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f%25.4f\n", 2 * size * sizeof(float)* 1e-6 / average_milliseconds2, average_milliseconds2 );


    /*  -----------------------  */

    printf("%25s","Navie Transpose");

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    float total_milliseconds3 = 0;
    for (int i=0; i < num_runs; i++){
        cudaEventRecord(start3);
        navie_transpose<<<gridSize, blockSize>>>(d_data, d_output, rows);
        cudaEventRecord(stop3);
        cudaEventSynchronize(stop3);
        float milliseconds3 = 0;
        cudaEventElapsedTime(&milliseconds3, start3, stop3);
        total_milliseconds3 += milliseconds3;
    }
    float average_milliseconds3 = total_milliseconds3 / num_runs;
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    cudaMemcpy(c_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f%25.4f\n", 2 * size * sizeof(float)* 1e-6 / average_milliseconds3, average_milliseconds3 );


    /*  -----------------------  */

    printf("%25s","Shared Coalesced");

    cudaEvent_t start4, stop4;
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);
    float total_milliseconds4 = 0;
    for (int i=0; i < num_runs; i++){
        cudaEventRecord(start4);
        shared_coalesced_bank_conflict<<<gridSize, blockSize>>>(d_data, d_output, rows);
        cudaEventRecord(stop4);
        cudaEventSynchronize(stop4);
        float milliseconds4 = 0;
        cudaEventElapsedTime(&milliseconds4, start4, stop4);
        total_milliseconds4 += milliseconds4;
    }
    float average_milliseconds4 = total_milliseconds4 / num_runs;
    cudaEventDestroy(start4);
    cudaEventDestroy(stop4);
    cudaMemcpy(c_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f%25.4f\n", 2 * size * sizeof(float)* 1e-6 / average_milliseconds4, average_milliseconds4 );



    /*  -----------------------  */

    printf("%25s","No Bank Conflict");

    cudaEvent_t start5, stop5;
    cudaEventCreate(&start5);
    cudaEventCreate(&stop5);
    float total_milliseconds5 = 0;
    for (int i=0; i < num_runs; i++){
        cudaEventRecord(start5);
        shared_coalesced_no_bank_conflict<<<gridSize, blockSize>>>(d_data, d_output, rows);
        cudaEventRecord(stop5);
        cudaEventSynchronize(stop5);
        float milliseconds5 = 0;
        cudaEventElapsedTime(&milliseconds5, start5, stop5);
        total_milliseconds5 += milliseconds5;
    }
    float average_milliseconds5 = total_milliseconds5 / num_runs;
    cudaEventDestroy(start5);
    cudaEventDestroy(stop5);
    cudaMemcpy(c_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f%25.4f\n", 2 * size * sizeof(float)* 1e-6 / average_milliseconds5, average_milliseconds5 );



    // for (int row = 0; row < 10; row ++){
    //     for (int col = 0; col < 10; col++){
    //         std::cout << c_data[row*cols + col] << " " ;
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "\n\n";

    // for (int row = 0; row < 10; row ++){
    //     for (int col = 0; col < 10; col++){
    //         std::cout << c_output[row*cols + col] << " " ;
    //     }
    //     std::cout << "\n";
    // }
    
    // Free memory
    cudaFree(d_data);
    cudaFree(d_output);
    delete[] c_data;
    delete[] c_output;


    cudaDeviceReset(); // Important: Release CUDA resources
    return 0;
}
