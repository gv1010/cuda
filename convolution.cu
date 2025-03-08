#include <iostream>
#include <cuda_runtime.h>

#define filter_radius 2
#define tile_dim 32

__constant__ float hC[(2*filter_radius + 1)*(2*filter_radius + 1)];

__global__ void conv2d(float *inp, float *filt, float * oup, int rows, int cols){
    int gty = blockIdx.x * blockDim.x + threadIdx.x;
    int gtx = blockIdx.y * blockDim.y + threadIdx.y;

    float accumulate = 0;
    for (int fr = -1*filter_radius; fr < filter_radius+1; fr++){
        for (int fc = -1*filter_radius; fc < filter_radius+1; fc++){
            int gtx_new = gtx+fr;
            int gty_new = gty+fc;
            if (gtx_new >= 0 && gtx_new < rows && gty_new >= 0 && gty_new < cols){

                int abs_fr = fr + filter_radius;
                int abs_fc = fc + filter_radius;
                
                accumulate += filt[abs_fr * (2*filter_radius + 1) + abs_fc] * inp[gtx_new*cols + gty_new];
                
            }
        }
    oup[gtx*cols + gty] = accumulate;
    }
}


__global__ void conv2d_with_constant_memory(float *inp, float * oup, int rows, int cols){
    int gty = blockIdx.x * blockDim.x + threadIdx.x;
    int gtx = blockIdx.y * blockDim.y + threadIdx.y;

    float accumulate = 0;
    for (int fr = -1*filter_radius; fr < filter_radius+1; fr++){
        for (int fc = -1*filter_radius; fc < filter_radius+1; fc++){
            int gtx_new = gtx+fr;
            int gty_new = gty+fc;
            if (gtx_new >= 0 && gtx_new < rows && gty_new >= 0 && gty_new < cols){

                int abs_fr = fr + filter_radius;
                int abs_fc = fc + filter_radius;
                
                accumulate += hC[abs_fr * (2*filter_radius + 1) + abs_fc] * inp[gtx_new*cols + gty_new];
                
            }
        }
    oup[gtx*cols + gty] = accumulate;
    }
}


__global__ void conv2d_with_constant_memory_shared_memory(float *inp, float * oup, int rows, int cols){

    __shared__ int shared_tile[tile_dim][tile_dim+1];
    
    int gty = blockIdx.x * blockDim.x + threadIdx.x;
    int gtx = blockIdx.y * blockDim.y + threadIdx.y;

    shared_tile[threadIdx.y][threadIdx.x] = inp[gtx*rows + gty];
    __syncthreads();

    float accumulate = 0;
    for (int fr = -1*filter_radius; fr < filter_radius+1; fr++){
        for (int fc = -1*filter_radius; fc < filter_radius+1; fc++){
            // int gtx_new = gtx+fr;
            // int gty_new = gty+fc;
            // if (gtx_new >= 0 && gtx_new < rows && gty_new >= 0 && gty_new < cols){

            int abs_fr = fr + filter_radius;
            int abs_fc = fc + filter_radius;
            float tid_y_fc = threadIdx.y + fc;
            float tid_x_fr = threadIdx.x + fr;

            if (tid_y_fc >= 0 && tid_y_fc < tile_dim && tid_x_fr >= 0 && tid_x_fr < tile_dim ){
                accumulate += hC[abs_fr * (2*filter_radius + 1) + abs_fc] * shared_tile[threadIdx.y + fc][threadIdx.x + fr];
            }
                
        }
    oup[gtx*rows + gty] = accumulate;
    }
}

int main(){
    int rows = 1024;
    int cols = 1024;

    // int filter_radius = 2;

    float *hA, *hB, *hFilt;
    float *dA, *dB, *dC;

    hA = new float[rows*cols];
    hB = new float[rows*cols];
    hFilt = new float[(2*filter_radius + 1)*(2*filter_radius + 1)]; //filter

    cudaMalloc((void**)&dA,  rows*cols*sizeof(float));
    cudaMalloc((void**)&dB,  rows*cols*sizeof(float));
    cudaMalloc((void**)&dC,  rows*cols*sizeof(float));

    for (int i=0; i < rows; i++){
        for (int j=0; j < cols; j++){
            hA[i*cols + j] = 1;
        }
    }

    for (int i=0; i < 2*filter_radius + 1; i++){
        for (int j=0; j < 2*filter_radius + 1; j++){
            if (i == filter_radius || j == filter_radius){
                hFilt[i*(2*filter_radius + 1) + j] = 1;
            }
            else{
                hFilt[i*(2*filter_radius + 1) + j] = 0;
            }
        }
    }

    cudaMemcpy(dA, hA, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hFilt, (2*filter_radius + 1)*(2*filter_radius + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(hC, hFilt, (2*filter_radius + 1)*(2*filter_radius + 1) * sizeof(float));


    dim3 block_size(tile_dim, tile_dim);
    dim3 grid_size((cols + block_size.x - 1)/block_size.x, (rows + block_size.y - 1)/block_size.y);
    int num_runs = 100;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    float total_ms1 = 0;

    printf("%25s%25s\n", "Exectution", "GPU Time (ms)");
    printf("\n");

    printf("%25s", "Naive Conv 2D");

    for (int i=0; i< num_runs; i++){
        cudaEventRecord(start1);
        conv2d<<<grid_size, block_size>>>(dA, dC, dB, rows, cols);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start1, stop1);
        total_ms1 += milliseconds1;
    }
    float average_milliseconds1 = total_ms1 / num_runs;
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaMemcpy(hB, dB, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f", average_milliseconds1 );
    
    printf("\n");
    printf("%25s", "Constant Memory Conv 2D");
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float total_ms2 = 0;
    for (int i=0; i< num_runs; i++){
        cudaEventRecord(start2);
        conv2d_with_constant_memory<<<grid_size, block_size>>>(dA, dB, rows, cols);
        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        float milliseconds2 = 0;
        cudaEventElapsedTime(&milliseconds2, start2, stop2);
        total_ms2 += milliseconds2;
    }
    float average_milliseconds2 = total_ms2 / num_runs;
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaMemcpy(hB, dB, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%20.4f", average_milliseconds2 );


    printf("\n");
    printf("%25s", "Shared & Const Memory Conv 2D");
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    float total_ms3 = 0;
    for (int i=0; i< num_runs; i++){
        cudaEventRecord(start3);
        conv2d_with_constant_memory_shared_memory<<<grid_size, block_size>>>(dA, dB, rows, cols);
        cudaEventRecord(stop3);
        cudaEventSynchronize(stop3);
        float milliseconds3 = 0;
        cudaEventElapsedTime(&milliseconds3, start3, stop3);
        total_ms3 += milliseconds3;
    }
    float average_milliseconds3 = total_ms3 / num_runs;
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    cudaMemcpy(hB, dB, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%17.4f", average_milliseconds3 );



    // conv2d_with_constant_memory_shared_memory
    // printf("\nhB: \n");
    // for (int row = 0; row < 32; row ++){
    //     for (int col = 0; col < 32; col++){
    //         printf("%0.0f ", hB[row*cols + col]);
    //     }
    //     printf("\n");
    // }
    return 0;   
}
