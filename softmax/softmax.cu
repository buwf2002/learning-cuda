#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define NUM_WARP 16
#define BLOCK_SIZE (32 * NUM_WARP)

struct FMax {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct FAdd {
    __device__ float operator()(float a, float b) const { return a + b; }
};

template <typename Func>
__device__ void warpReduce(float* val, Func f) {
    for (int offset = 16; offset > 0; offset /= 2) {
        *val = f(*val, __shfl_down_sync(0xffffffff, *val, offset));
    }
}

template <typename Func>
__device__ void blockReduce(float* val, float* smem, Func f) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) smem[warp_id] = *val;
    __syncthreads();

    if (warp_id == 0) {
        *val = (lane < NUM_WARP) ? smem[lane] : 0;
        warpReduce(val, f);
    }
    __syncthreads();

    if (lane == 0 && warp_id == 0) smem[0] = *val;
    __syncthreads();
    *val = smem[0];
}


template <typename Func>
__device__ void blockReduce_(float* val, float* smem, Func f) {

    smem[threadIdx.x] = *val;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset >= 32; offset /=2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] = f(smem[threadIdx.x], smem[threadIdx.x + offset]);
        }
        __syncthreads();
    }
}

template <typename Func>
__device__ void warpReduce_(float* val, float *smem, Func f) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (warp_id == 0){
        *val = smem[lane];
        for (int offset = 16; offset > 0; offset /= 2) {
            *val = f(*val, __shfl_down_sync(0xffffffff, *val, offset));
        }
    }
}

// Naive softmax kernel using and shared memory reduction

__global__ void softmaxNaiveKernel(const float* __restrict__ input, float* __restrict__ output, int m, int n) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    if (row >= m) return;
    int start = row * n;

    __shared__ float smem[NUM_WARP];
    
    // Step 1: Find max value
    float max_val = -FLT_MAX;
    for (int i = idx; i < n; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, input[start + i]);
    }
    warpReduce(&max_val, fmaxf);
    blockReduce(&max_val, smem, fmaxf);
    
    // Step 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = idx; i < n; i += BLOCK_SIZE) {
        sum += __expf(input[start + i] - max_val);
    }
    warpReduce(&sum, __fadd_rn);
    blockReduce(&sum, smem, __fadd_rn);
    
    // Step 3: Normalize
    for (int i = idx; i < n; i += BLOCK_SIZE) {
        output[start + i] = __expf(input[start + i] - max_val) / sum;
    }
}

void softmaxNaiveExec(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid(m);  // One block per row
    softmaxNaiveKernel<<<grid, block>>>(input, output, m, n);
    cudaDeviceSynchronize();
}

// Vectorized version using float4

__global__ void softmaxVec4Kernel(const float* __restrict__ input, float* __restrict__ output, int m, int n) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    if (row >= m) return;

    input += row * n;
    output += row * n;
    __shared__ float smem[NUM_WARP];

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
    blockReduce(&max_val, smem, fmaxf);
    
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
    blockReduce(&sum, smem, __fadd_rn);
    
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

void softmaxVec4Exec(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid(m);  // One block per row
    softmaxVec4Kernel<<<grid, block>>>(input, output, m, n);
    cudaDeviceSynchronize();
}

__global__ void softmaxVec4SmemTreeKernel(const float* __restrict__ input, float* __restrict__ output, int m, int n) {
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
    blockReduce_(&max_val, smem, fmaxf);
    warpReduce_(&max_val, smem, fmaxf);
    // thread 0 has the max value, broadcast to shared memory for other threads
    if (threadIdx.x == 0) smem[0] = max_val;
    __syncthreads();
    max_val = smem[0];
    
    // Step 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = idx; i < n / 4; i += BLOCK_SIZE) {
        float4 val = *reinterpret_cast<const float4*>(input + i * 4);
        sum += __expf(val.x - max_val);
        sum += __expf(val.y - max_val);
        sum += __expf(val.z - max_val);
        sum += __expf(val.w - max_val);
    }
    blockReduce_(&sum, smem, __fadd_rn);
    warpReduce_(&sum, smem, __fadd_rn);
    // thread 0 has the sum value, broadcast to shared memory for other threads
    if (threadIdx.x == 0) smem[0] = sum;
    __syncthreads();
    sum = smem[0];
    
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

void softmaxVec4SmemTreeExec(const float* __restrict__ input, float* __restrict__ output, int m, int n){
    dim3 block(BLOCK_SIZE);
    dim3 grid(m);  // One block per row
    softmaxVec4SmemTreeKernel<<<grid, block>>>(input, output, m, n);
    cudaDeviceSynchronize();
}