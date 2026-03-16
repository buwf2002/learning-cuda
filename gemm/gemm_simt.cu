#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

template<int BM, int BN>
__global__ void gemm_kernel_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float * __restrict__ C,
    int m, int n, int k
){
    int n_idx = blockIdx.x * BN + threadIdx.x;
    int m_idx = blockIdx.y * BM + threadIdx.y;

    if (m_idx >= m || n_idx >= n)
        return;

    float sum = 0;
    for (int i = 0; i < k; i++){
        sum += A[m_idx * k + i] * B[i * n + n_idx];
    }

    C[m_idx * n + n_idx] = sum;
}

void gemm_naive(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float * __restrict__ C, 
    int m, int n, int k
){
    const int BM = 32;
    const int BN = 32;
    // const int BLK = BM * BN;
    dim3 block(BM, BN);
    dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
    gemm_kernel_naive<BM, BN><<<grid, block>>>(A, B, C, m, n, k);
}

template<int BM, int BN, int BK>
__global__ void gemm_kernel_block_tile(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float * __restrict__ C,
    int m, int n, int k
){
    int n_idx = blockIdx.x * BN;
    int m_idx = blockIdx.y * BM;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int k_iter = (k + BK - 1) / BK;

    float sum = 0;
    #pragma unroll
    for(int i = 0; i < k_iter; i++){
        // load A: row-major, tile [BM][BK], stride k
        As[threadIdx.y][threadIdx.x] = (m_idx + threadIdx.y < m && i * BK + threadIdx.x < k)
            ? A[(m_idx + threadIdx.y) * k + (i * BK + threadIdx.x)] : 0;
        // load B: row-major, tile [BK][BN], stride n
        Bs[threadIdx.y][threadIdx.x] = (i * BK + threadIdx.y < k && n_idx + threadIdx.x < n)
            ? B[(i * BK + threadIdx.y) * n + (n_idx + threadIdx.x)] : 0;
        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BK; j++){
            sum += As[threadIdx.y][j] * Bs[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (m_idx + threadIdx.y < m && n_idx + threadIdx.x < n) {
        C[(m_idx + threadIdx.y) * n + (n_idx + threadIdx.x)] = sum;
    }
}

void gemm_block_tile(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float * __restrict__ C, 
    int m, int n, int k
){
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    // const int BLK = BM * BN;
    dim3 block(BM, BN);
    dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
    gemm_kernel_block_tile<BM, BN, BK><<<grid, block>>>(A, B, C, m, n, k);
}

template<int BM, int BN, int BK>
__global__ void gemm_kernel_block_tile_double_buffer(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float * __restrict__ C,
    int m, int n, int k
){
    int n_idx = blockIdx.x * BN;
    int m_idx = blockIdx.y * BM;

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    const int k_iter = (k + BK - 1) / BK;
    float sum = 0;

    auto load_a_smem = [&](int k_idx, int buffer_idx)
    {
        As[buffer_idx][threadIdx.y][threadIdx.x] = (m_idx + threadIdx.y < m && k_idx * BK + threadIdx.x < k)
            ? A[(m_idx + threadIdx.y) * k + (k_idx * BK + threadIdx.x)] : 0;
    };

    auto load_b_smem = [&](int k_idx, int buffer_idx)
    {
        Bs[buffer_idx][threadIdx.y][threadIdx.x] = (k_idx * BK + threadIdx.y < k && n_idx + threadIdx.x < n)
            ? B[(k_idx * BK + threadIdx.y) * n + (n_idx + threadIdx.x)] : 0;
    };

    auto compute = [&](int buffer_idx)
    {
#pragma unroll
        for(int j = 0; j < BK; j++)
        {
            sum += As[buffer_idx][threadIdx.y][j] * Bs[buffer_idx][j][threadIdx.x];
        }
    };

    load_a_smem(0, 0);
    load_b_smem(0, 0);

#pragma unroll
    for(int i = 0; i < k_iter; i++){
        int cur_buffer_idx = i % 2;
        int next_buffer_idx = (i + 1) % 2;

        __syncthreads();
        load_a_smem(i + 1, next_buffer_idx);
        load_b_smem(i + 1, next_buffer_idx);

        compute(cur_buffer_idx);
    }

    if (m_idx + threadIdx.y < m && n_idx + threadIdx.x < n) {
        C[(m_idx + threadIdx.y) * n + (n_idx + threadIdx.x)] = sum;
    }
}

void gemm_block_tile_double_buffer(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float * __restrict__ C, 
    int m, int n, int k
){
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    // const int BLK = BM * BN;
    dim3 block(BM, BN);
    dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
    gemm_kernel_block_tile_double_buffer<BM, BN, BK><<<grid, block>>>(A, B, C, m, n, k);
}
