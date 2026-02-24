#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define NUM_WARPS 8

// Naive reduction kernel using global atomicAdd
template <typename T>
__global__ void reduceKernelNaive(const T* d_input, T* d_output, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = 0;

#pragma unroll
    for (size_t i = idx; i < N; i += blockDim.x * gridDim.x) {
        val += d_input[i];
    }

    atomicAdd(d_output, val); 
}

template <typename T>
cudaError_t reduceExecuteNaive(const T* d_input, T* d_output, size_t N) {
    int threads = NUM_WARPS * 32; // 每个块内的线程数
    int blocks = (N + threads - 1) / threads;
    
    reduceKernelNaive<T><<<blocks, threads>>>(d_input, d_output, N);
    
    return cudaGetLastError();
}

// Block-level reduction kernel
template <typename T>
__global__ void reduceKernelBlockReduce(const T* d_input, T* d_output, size_t N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T block_sum = 0;
    if (threadIdx.x == 0){

        #pragma unroll
        for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
            #pragma unroll
            for (int j = 0; j < blockDim.x; ++j) {
                if (i + j < N){
                    block_sum += d_input[i + j];
                }
            }
        }
        atomicAdd(d_output, block_sum); // Atomically add the block's sum to the output
    }
}

template <typename T>
cudaError_t reduceExecuteBlockReduce(const T* d_input, T* d_output, size_t N) {
    int threads = 32 * NUM_WARPS;
    int blocks = (N + threads - 1) / threads;
    reduceKernelBlockReduce<T><<<blocks, threads>>>(d_input, d_output, N);
    return cudaGetLastError();
}

// Warp-level reduction kernel
template <typename T>
__global__ void reduceKernelWarpReduce(const T* d_input, T* d_output, size_t N) {

    int lane_id = threadIdx.x % 32; // 当前线程在 warp 中的 lane ID
    int warp_id = threadIdx.x / 32; // 当前线程所在的 warp ID
    int idx = blockIdx.x * blockDim.x + warp_id * 32; // 每个 warp 处理一个连续的 32 元素块

    T warp_sum = 0;
    if (lane_id == 0){
        #pragma unroll
        for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                if (i + j < N){
                    warp_sum += d_input[i + j];
                }
            }

        }
        
        atomicAdd(d_output, warp_sum); // Atomically add the warp's sum to the output
    }
}

template <typename T>
cudaError_t reduceExecuteWarpReduce(const T* d_input, T* d_output, size_t N) {
    int threads = 32 * NUM_WARPS;
    int blocks = (N + threads - 1) / threads;
    reduceKernelWarpReduce<T><<<blocks, threads>>>(d_input, d_output, N);
    return cudaGetLastError();
}

// Block-level reduction kernel using shared memory
template <typename T>
__global__ void reduceKernelSmemReduce(const T* d_input, T* d_output, size_t N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T smem[32 * NUM_WARPS]; // 每个线程一个位置的共享内存
    smem[threadIdx.x] = 0; // 初始化共享内存

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        smem[threadIdx.x] += d_input[i];
    }
    __syncthreads(); // 确保所有线程都完成了数据加载

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        }
        __syncthreads(); // 确保所有线程都完成了当前步
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_output, smem[0]);
    }
}

template <typename T>
cudaError_t reduceExecuteSmemReduce(const T* d_input, T* d_output, size_t N) {
    int threads = 32 * NUM_WARPS;
    int blocks = (N + threads - 1) / threads;
    reduceKernelSmemReduce<T><<<blocks, threads>>>(d_input, d_output, N);
    return cudaGetLastError();
}

// Block-level reduction kernel using shft instructions
template <typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__global__ void reduceKernelShflReduce(const T* d_input, T* d_output, size_t N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32; // 当前线程在 warp 中的 lane ID
    int warp_id = threadIdx.x / 32; // 当前线程所在的 warp ID

    __shared__ T smem[NUM_WARPS]; // 每个线程一个位置的共享内存
    if (lane_id == 0) smem[warp_id] = 0; // 初始化共享内存

    T val = 0;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        val += d_input[i];
    }
    val = warpReduceSum(val); // 在 warp 内进行归约

    // warp 之间进行规约
    // 1.warp内部的reduce结果写入smem 2. 每个warp的第一个线程读取smem进行全局规约
    if (lane_id == 0){
        smem[warp_id] = val; // 将每个 warp 的结果写入共享内存
    }
    __syncthreads(); // 确保所有 warp 都完成了写入
    if (warp_id == 0){
        val = (threadIdx.x < NUM_WARPS) ? smem[lane_id] : 0; // 只有前 NUM_WARPS 个线程参与规约
        val = warpReduceSum(val); // 在第一个 warp 内进行规约
        if (lane_id == 0) {
            atomicAdd(d_output, val); // 将最终结果写入全局内存
        }
    }
}

template <typename T>
cudaError_t reduceExecuteShflReduce(const T* d_input, T* d_output, size_t N) {
    int threads = 32 * NUM_WARPS;
    int blocks = (N + threads - 1) / threads;
    reduceKernelShflReduce<T><<<blocks, threads>>>(d_input, d_output, N);
    return cudaGetLastError();
}

