#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define NUM_WARPS 8
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS) // 512 threads per block

template <typename Func>
__device__ void blockReduceSBR(float* val, float* smem, Func f) {
    smem[threadIdx.x] = *val;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] = f(smem[threadIdx.x], smem[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    *val = smem[0];
}

template <typename Func>
__device__ void warpReduce(float* val, Func f) {
    for (int offset = 16; offset > 0; offset /= 2) {
        *val = f(*val, __shfl_down_sync(0xffffffff, *val, offset));
    }
}

// Naive softmax kernel using and shared memory reduction

__global__ void softmax_kernel_sbr(const float* __restrict__ input, float* __restrict__ output, int m, int n) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    if (row >= m) return;
    int start = row * n;

    __shared__ float smem[BLOCK_SIZE];

    // Step 1: Find max value
    float max_val = -FLT_MAX;
    for (int i = idx; i < n; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, input[start + i]);
    }
    blockReduceSBR(&max_val, smem, fmaxf);
    __syncthreads();
    max_val = smem[0];

    // Step 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = idx; i < n; i += BLOCK_SIZE) {
        sum += __expf(input[start + i] - max_val);
    }
    blockReduceSBR(&sum, smem, __fadd_rn);
    __syncthreads();
    sum = smem[0];

    // Step 3: Normalize
    for (int i = idx; i < n; i += BLOCK_SIZE) {
        output[start + i] = __expf(input[start + i] - max_val) / sum;
    }
}

void softmax_exec_sbr(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid(m);
    softmax_kernel_sbr<<<grid, block>>>(input, output, m, n);
}

// Vectorized version using float4

__global__ void softmax_kernel_sbr_v4(const float* __restrict__ input, float* __restrict__ output, int m, int n) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    if (row >= m) return;

    input += row * n;
    output += row * n;
    __shared__ float smem[BLOCK_SIZE];

    // Step 1: Find max value
    float max_val = -FLT_MAX;
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        max_val = fmaxf(max_val, val.x);
        max_val = fmaxf(max_val, val.y);
        max_val = fmaxf(max_val, val.z);
        max_val = fmaxf(max_val, val.w);
    }
    blockReduceSBR(&max_val, smem, fmaxf);

    // Step 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        sum += __expf(val.x - max_val);
        sum += __expf(val.y - max_val);
        sum += __expf(val.z - max_val);
        sum += __expf(val.w - max_val);
    }
    blockReduceSBR(&sum, smem, __fadd_rn);

    // Step 3: Normalize
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        float4 out;
        out.x = __expf(val.x - max_val) / sum;
        out.y = __expf(val.y - max_val) / sum;
        out.z = __expf(val.z - max_val) / sum;
        out.w = __expf(val.w - max_val) / sum;
        *reinterpret_cast<float4*>(output + i * 4) = out;
    }
}

void softmax_exec_sbr_v4(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid(m);
    softmax_kernel_sbr_v4<<<grid, block>>>(input, output, m, n);
    // cudaDeviceSynchronize();
}

// 1. warp reduce 2. block reduce 3. broadcast max and sum to all threads in block using shared memory tree reduction
template <typename Func>
__device__ void blockReduceIntraWarp(float* val, float* smem, Func f) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        smem[wid] = *val;
    }
    __syncthreads();

    if (wid == 0){
        *val = (lane < NUM_WARPS) ? smem[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            *val = f(*val, __shfl_down_sync(0xffffffff, *val, offset));
        }
    }
    // broadcast the result to all threads in the block
    if (threadIdx.x == 0) {
        smem[0] = *val;
    }
    __syncthreads();
    *val = smem[0];
}
__global__ void softmax_kernel_wr_v4(const float* __restrict__ input, float* __restrict__ output, int m, int n) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    if (row >= m) return;

    input += row * n;
    output += row * n;
    __shared__ float smem[NUM_WARPS];

    // Step 1: Find max value
    float max_val = -FLT_MAX;
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        max_val = fmaxf(max_val, val.x);
        max_val = fmaxf(max_val, val.y);
        max_val = fmaxf(max_val, val.z);
        max_val = fmaxf(max_val, val.w);
    }
    warpReduce(&max_val, fmaxf);
    blockReduceIntraWarp(&max_val, smem, fmaxf);

    // Step 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        sum += __expf(val.x - max_val);
        sum += __expf(val.y - max_val);
        sum += __expf(val.z - max_val);
        sum += __expf(val.w - max_val);
    }
    warpReduce(&sum, __fadd_rn);
    blockReduceIntraWarp(&sum, smem, __fadd_rn);

    // Step 3: Normalize
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        float4 out;
        out.x = __expf(val.x - max_val) / sum;
        out.y = __expf(val.y - max_val) / sum;
        out.z = __expf(val.z - max_val) / sum;
        out.w = __expf(val.w - max_val) / sum;
        *reinterpret_cast<float4*>(output + i * 4) = out;
    }
}

void softmax_exec_wr_v4(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid(m);
    softmax_kernel_wr_v4<<<grid, block>>>(input, output, m, n);
    // cudaDeviceSynchronize();
}

template <typename Func>
__device__ void blockReduce2warp(float* val, float* smem, Func f) {
    smem[threadIdx.x] = *val;
    __syncthreads();
    for(int offset = BLOCK_SIZE / 2; offset >= 32; offset /= 2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] = f(smem[threadIdx.x], smem[threadIdx.x + offset]);
        }
        __syncthreads();
    }
}

template <typename Func>
__device__ void warpReduceSmem(float* val, float* smem, Func f) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (threadIdx.x < WARP_SIZE) {
        *val = smem[threadIdx.x];
    }
    __syncthreads();

    if (wid == 0){
        for (int offset = 16; offset > 0; offset /= 2) {
            *val = f(*val, __shfl_down_sync(0xffffffff, *val, offset));
        }
    }

    if (threadIdx.x == 0) {
        smem[0] = *val;
    }
    __syncthreads();
    *val = smem[0];
}
__global__ void softmax_kernel_sbr_wr_v4(const float* __restrict__ input, float* __restrict__ output, int m, int n) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    if (row >= m) return;

    input += row * n;
    output += row * n;
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = 0.0f; // 初始化共享内存
    __syncthreads();

    // Step 1: Find max value
    float max_val = -FLT_MAX;
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        max_val = fmaxf(max_val, val.x);
        max_val = fmaxf(max_val, val.y);
        max_val = fmaxf(max_val, val.z);
        max_val = fmaxf(max_val, val.w);
    }
    blockReduce2warp(&max_val, smem, fmaxf);
    warpReduceSmem(&max_val, smem, fmaxf);

    // Step 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        sum += __expf(val.x - max_val);
        sum += __expf(val.y - max_val);
        sum += __expf(val.z - max_val);
        sum += __expf(val.w - max_val);
    }
    blockReduce2warp(&sum, smem, __fadd_rn);
    warpReduceSmem(&sum, smem, __fadd_rn);

    // Step 3: Normalize
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        float4 out;
        out.x = __expf(val.x - max_val) / sum;
        out.y = __expf(val.y - max_val) / sum;
        out.z = __expf(val.z - max_val) / sum;
        out.w = __expf(val.w - max_val) / sum;
        *reinterpret_cast<float4*>(output + i * 4) = out;
    }
}

void softmax_exec_sbr_wr_v4(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid(m);
    softmax_kernel_sbr_wr_v4<<<grid, block>>>(input, output, m, n);
    // cudaDeviceSynchronize();
}