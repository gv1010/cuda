# cuda

## Pinned meory
### usage of pinned memory
    - the data is copied to pinned memory and then copied to gpu memory
    - using pinned memory on host using the command cudaMallocHost(),  cudaHostAlloc(), and deallocate it with cudaFreeHost(). 
    - the pinned memory is copied to GPU, 
    - should not over-allocate pinned memory. Doing so can reduce overall system performance because it reduces the amount of physical memory available to the operating system and other programs.
    - Due to the overhead associated with each transfer, it is preferable to batch many small transfers together into a single transfer. This is easy to do by using a temporary array, preferably pinned, and packing it with the data to be transferred.

    - For two-dimensional array transfers, you can use cudaMemcpy2D(). cudaMemcpy2D(dest, dest_pitch, src, src_pitch, w, h, cudaMemcpyHostToDevice)
    - use nvprof, the command-line CUDA profiler
    
    **  nvcc pin_memory.cu -o pin_memory
      nvprof ./pin_memory **
