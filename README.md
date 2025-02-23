# cuda

**Pinned Memory**
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

**CUDA Streams**

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









