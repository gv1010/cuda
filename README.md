# CUDA

### Pinned Memory
[source](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
*   the data is copied to pinned memory and then copied to gpu memory
*   using pinned memory on host using the command cudaMallocHost(),  cudaHostAlloc(), and deallocate it with cudaFreeHost(). 
*   the pinned memory is copied to GPU, 
*   should not over-allocate pinned memory. Doing so can reduce overall system performance because it reduces the amount of physical memory available to the operating system and other programs.
*   Due to the overhead associated with each transfer, it is preferable to batch many small transfers together into a single transfer. This is easy to do by using a temporary array, preferably pinned, and packing it with the data to be transferred.
*   For two-dimensional array transfers, you can use cudaMemcpy2D(). cudaMemcpy2D(dest, dest_pitch, src, src_pitch, w, h, cudaMemcpyHostToDevice)
*   use nvprof, the command-line CUDA profiler
        ```
        nvcc pin_memory.cu -o pin_memory
        nvprof ./pin_memory
        ```

### CUDA Streams

[source](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)

*   sequence of operations that execute on the device in the order, given by the host code.
*   Within a stream the operations are garunateed to be executed in an order, but operations in different streams can be interleaved or may run concurrently.
*   **default stream :** all device operation are run in a stream, default stream is different from other streams as this is a synchronizing stream, as no operation on the default stream begin until all previously issued operations in any stream on the device are completed and an opearation in the default stream must complete before any other operation (in any stream on the device) will begin.

* the code from the host prespective, executes synchronously, after calling the kernel, the host code moves to the next line, where the d2h transfer will happen only after the device-side order of executions. due the async behaviour from the host after the kernel is invoked makes overlapping device and host computation

    ```
    cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
    increment<<<1,N>>>(d_a)
    cudaMemcpy(a, d_a, numBytes, cudaMemcpyDeviceToHost);
    ```

*   Non Default Stream:  cudaMemcpyAsync() is non-blocking on the host, so control returns to the host thread immediately after the transfer is issued. There are cudaMemcpy2DAsync() and cudaMemcpy3DAsync() variants of this routine which can transfer 2D and 3D array sections asynchronously in the specified streams.
*   Since all operations in non-default streams are non-blocking with respect to the host code, you will run across situations where you need to synchronize the host code with operations in a stream.
        - cudaDeviceSynchronize() reduces perfromance
        - cudaStreamSynchronize(stream)

*   We want to achieve overlapping kernel exection and data transfers.
*   we do this Async on different streams
  


### Convolution
[**Source Code (convolution.cu)**](convolution.cu)
1. **Naive Convolution**
   -  a simple element-wise multiplication and summation within the filter window. 
   -  global memory access of the filter elements and input matrix
2. **Convolution with constant memory**
   -  here we are creating a constant memory, it is a special, read-only memory space on the GPU designed for storing data that remains constant during the execution of a kernel.
   -  used to store data that is read by many threads but does not change during the kernel's execution. This makes it good for storing coefficients, lookup tables, or other fixed parameters.
   -  Constant memory is cached on the GPU, which can significantly improve performance when all threads (or threads within a warp) access the same memory location.   
      Once initialized by the host (CPU), the GPU threads can only read from constant memory. There's a limited amount of constant memory available (typically 64KB).
   -  Still we are performing global memory access of the input element. This can be further optimized with shared memory where a tile in a block can save elements which can be read by all the threads in a block.
3. **Shared and Constant Memory**
   -  Shared memory is used to reduce redundant global loads by reusing data, this reside on the GPU.
   -  Here the difference in execution is not significant in comparison to filter constant memory execution. one of the reason may be due to __syncthreads(); which holds the thread operations until all threads in a block finishes exectuion until 
      each thread in a block reaches __syncthreads(); there are other factors like tile/filter size, persistent memory bandwidth limits, and implementation efficiency.

        

     
      ```
                     Exectution            GPU Time (ms)
        
                    Naive Conv 2D              0.2227
          Constant Memory Conv 2D              0.1699
        Shared & Const Memory Conv 2D          0.1478


        ==949== NVPROF is profiling process 949, command: ./t1
        ==949== Profiling application: ./t1
        ==949== Profiling result:
                    Type  Time(%)      Time     Calls       Avg       Min       Max  Name
         GPU activities:   36.79%  21.696ms       100  216.96us  216.35us  217.50us  conv2d(float*, float*, float*, int, int)
                           27.83%  16.415ms       100  164.15us  163.01us  165.47us  conv2d_with_constant_memory(float*, float*, int, int)
                           24.06%  14.187ms       100  141.87us  138.53us  144.38us  conv2d_with_constant_memory_shared_memory(float*, float*, int, int)
                            7.00%  4.1282ms         3  1.3761ms  606.55us  2.9037ms  [CUDA memcpy DtoH]
                            4.32%  2.5481ms         4  637.03us     672ns  1.8039ms  [CUDA memcpy HtoD]

      ```






