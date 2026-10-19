// Lets a host C++ compiler compile and run the CUDA kernels gotranx generates, one thread
// at a time, so the tests can check them without a GPU (tests/test_cuda_kernels.py).
//
// The qualifiers become nothing, and blockIdx, blockDim and threadIdx are globals that
// gotranx_launch sets before calling the kernel once per thread.
#pragma once

#define __global__
#define __device__
#define __host__

struct gotranx_dim3 {
    unsigned int x = 0;
};

inline gotranx_dim3 blockIdx, blockDim, threadIdx;

// Run `kernel` as `kernel<<<blocks, threads>>>` would, sequentially
template <typename Kernel>
void gotranx_launch(long long blocks, unsigned int threads, Kernel kernel)
{
    blockDim.x = threads;
    for (long long block = 0; block < blocks; ++block) {
        blockIdx.x = static_cast<unsigned int>(block);
        for (unsigned int thread = 0; thread < threads; ++thread) {
            threadIdx.x = thread;
            kernel();
        }
    }
}
