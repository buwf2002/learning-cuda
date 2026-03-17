#include <cuda_fp16.h>
#include <math_constants.h>

// ============================================================================
// Sigmoid Implementation
// ============================================================================

// Hint: Each thread handles one element
// Formula: sigmoid(x) = 1.0f / (1.0f + expf(-x))
__global__ void sigmoid_naive_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < numel; i += blockDim.x * gridDim.x){
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

void sigmoid_exec_naive(const float* input, float* output, int numel) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_naive_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, numel);
}

// Hint: Load/store 4 floats at a time using float4
__global__ void sigmoid_vec4_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx * 4; i < numel; i += blockDim.x * gridDim.x * 4){
        float4 reg = *reinterpret_cast<const float4*>(input + i);
        reg.x = 1.0f / (1.0f + expf(-reg.x));
        reg.y = 1.0f / (1.0f + expf(-reg.y));
        reg.z = 1.0f / (1.0f + expf(-reg.z));
        reg.w = 1.0f / (1.0f + expf(-reg.w));

        *reinterpret_cast<float4*>(output + i) = reg;
    }
}

void sigmoid_exec_vec4(const float* input, float* output, int numel) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_vec4_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, numel);
}

// ============================================================================
// ReLU Implementation
// ============================================================================

__global__ void relu_naive_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < numel; i += blockDim.x * gridDim.x){
        output[i] = fmax(input[i], 0);
    }
}

void relu_exec_naive(const float* input, float* output, int numel) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_naive_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, numel);
}

__global__ void relu_vec4_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx * 4; i < numel; i += blockDim.x * gridDim.x * 4){
        float4 reg = *reinterpret_cast<const float4*>(input + i);
        reg.x = fmax(reg.x, 0);
        reg.y = fmax(reg.y, 0);
        reg.z = fmax(reg.z, 0);
        reg.w = fmax(reg.w, 0);

        *reinterpret_cast<float4*>(output + i) = reg;
    }
}

void relu_exec_vec4(const float* input, float* output, int numel) {
    const int BLOCK_SIZE = 256;
    int num_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_vec4_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, numel);
}

// ============================================================================
// LayerNorm Implementation
// ============================================================================

// TODO: Implement naive LayerNorm kernel
// Formula: y = (x - μ) / √(σ² + ε) * γ + β
// For simplicity, assume γ=1, β=0 (no learnable parameters)
// Each block handles one row (N elements)
template<int BLOCK_SIZE>
__global__ void layernorm_naive_kernel(const float* input, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    int idx = threadIdx.x;

    input += row * N;
    output += row * N;
    
    // Step 1: Compute mean
    __shared__ float mean;
    __shared__ float variance;
    __shared__ float r_std;
    __shared__ float smem[BLOCK_SIZE];

    smem[idx] = 0;
    __syncthreads();

    auto block_reduce_smem_sum = [&](){
        for(int i = idx; i < N; i += BLOCK_SIZE){ // 注意 i = idx
            smem[idx] += input[i];
        }
        __syncthreads();
        for(int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2){
            if (idx < offset){
                smem[idx] += smem[idx + offset];
            }
            __syncthreads();
        }
    };
    
    block_reduce_smem_sum();
    if (idx == 0) {
        mean = smem[0] / N;
    }
    __syncthreads();

    // Step 2: Compute variance
    smem[idx] = 0;
    __syncthreads();

    auto block_reduce_smem_variance = [&](){
        for(int i = idx; i < N; i += BLOCK_SIZE){
            smem[idx] += (input[i] - mean) * (input[i] - mean);
        }
        __syncthreads();
        for(int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2){
            if (idx < offset){
                smem[idx] += smem[idx + offset];
            }
            __syncthreads();
        }
    };

    block_reduce_smem_variance();
    
    if (idx == 0){
        variance = smem[0] / N;
        r_std = rsqrtf(variance +  eps);
    }
    __syncthreads();

    // Step 3: Normalize
    for(int i = idx; i < N; i += BLOCK_SIZE){
        output[i] = (input[i] - mean) * r_std;
    }
}

void layernorm_exec_naive(const float* input, float* output, int M, int N, float eps) {
    // Each block handles one row
    const int BLOCK_SIZE = 256;
    layernorm_naive_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE>>>(input, output, M, N, eps);
}

template<int BLOCK_SIZE>
__global__ void layernorm_vec4_kernel(const float* input, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    int idx = threadIdx.x;

    input += row * N;
    output += row * N;
    
    // Step 1: Compute mean
    __shared__ float mean;
    __shared__ float variance;
    __shared__ float r_std;
    __shared__ float smem[BLOCK_SIZE];

    smem[idx] = 0;
    __syncthreads();

    auto block_reduce_smem_sum = [&](){
        for(int i = idx * 4; i < N; i += BLOCK_SIZE * 4){ // 注意 i = idx
            float4 reg = *reinterpret_cast<const float4*>(input + i);
            smem[idx] += (reg.x + reg.y + reg.z + reg.w);
        }
        __syncthreads();
        for(int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2){
            if (idx < offset){
                smem[idx] += smem[idx + offset];
            }
            __syncthreads();
        }
    };
    
    block_reduce_smem_sum();
    if (idx == 0) {
        mean = smem[0] / N;
    }
    __syncthreads();

    // Step 2: Compute variance
    smem[idx] = 0;
    __syncthreads();

    auto block_reduce_smem_variance = [&](){
        for(int i = idx * 4; i < N; i += BLOCK_SIZE * 4){
            float4 reg = *reinterpret_cast<const float4*>(input + i);
            smem[idx] += (reg.x - mean) * (reg.x - mean);
            smem[idx] += (reg.y - mean) * (reg.y - mean);
            smem[idx] += (reg.z - mean) * (reg.z - mean);
            smem[idx] += (reg.w - mean) * (reg.w - mean);
        }
        __syncthreads();
        for(int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2){
            if (idx < offset){
                smem[idx] += smem[idx + offset];
            }
            __syncthreads();
        }
    };

    block_reduce_smem_variance();
    
    if (idx == 0){
        variance = smem[0] / N;
        r_std = rsqrtf(variance +  eps);
    }
    __syncthreads();

    // Step 3: Normalize
    for(int i = idx * 4; i < N; i += BLOCK_SIZE * 4){
        float4 reg = *reinterpret_cast<const float4*>(input + i);
        reg.x = (reg.x - mean) * r_std;
        reg.y = (reg.y - mean) * r_std;
        reg.z = (reg.z - mean) * r_std;
        reg.w = (reg.w - mean) * r_std;
        *(reinterpret_cast<float4*>(output + i)) = reg;
    }
}

void layernorm_exec_vec4(const float* input, float* output, int M, int N, float eps) {
    // Each block handles one row
    const int BLOCK_SIZE = 256;
    layernorm_vec4_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE>>>(input, output, M, N, eps);
}
