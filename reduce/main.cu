#include "src/reduce.cuh"

// 错误检查宏
#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} \

template <typename T>
using ReduceFunc = cudaError_t (*)(const T*, T*, size_t);

template <typename T>
void benchmark(ReduceFunc<T> func, const char* kernel_name, 
                     const T* d_input, T* d_output, size_t N, T expected_result) {
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int warmup_iters = 10;
    int test_iters = 100;
    size_t bytes = N * sizeof(T);

    // --- Warmup ---
    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUDA(cudaMemset(d_output, 0, sizeof(T)));
        CHECK_CUDA(func(d_input, d_output, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize()); 

    // --- Benchmark ---
    CHECK_CUDA(cudaEventRecord(start)); 
    for (int i = 0; i < test_iters; i++) {
        CHECK_CUDA(cudaMemset(d_output, 0, sizeof(T)));
        CHECK_CUDA(func(d_input, d_output, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop)); 

    // --- Statistics ---
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avg_ms = milliseconds / test_iters;
    float bandwidth = (bytes / 1e6) / avg_ms; // GB/s

    // --- Verification ---
    T h_output = 0;
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost));

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Kernel:                     " << kernel_name << std::endl;
    std::cout << "Average Time per iteration: " << avg_ms << " ms" << std::endl;
    std::cout << "Effective Bandwidth:        " << bandwidth << " GB/s" << std::endl;
    std::cout << "Expected Result:            " << expected_result << std::endl;
    std::cout << "Actual Result:              " << h_output << std::endl;
    
    // 兼容 int 和 float 的容差检查
    if (std::abs(static_cast<float>(h_output - expected_result)) < 1e-1) {
        std::cout << "Status:                     PASSED ✅" << std::endl;
    } else {
        std::cout << "Status:                     FAILED ❌" << std::endl;
    }
    std::cout << "--------------------------------------\n" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    size_t N = 2 * 1024 * 1024;
    size_t bytes = N * sizeof(float);

    // 1. Host 端初始化
    std::vector<float> h_input(N, 1.0f); // 全填 1.0f，最后求和结果理应等于 N

    // 2. Device 端分配内存并拷贝数据
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // 3. 创建 CUDA Events 用于精确计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    benchmark<float>(reduceExecuteNaive<float>, "Naive (Global AtomicAdd)", 
                           d_input, d_output, N, (float)N);
    benchmark<float>(reduceExecuteWarpReduce<float>, "Warp Reduce", 
                           d_input, d_output, N, (float)N);
    benchmark<float>(reduceExecuteBlockReduce<float>, "Block Reduce", 
                           d_input, d_output, N, (float)N);
    benchmark<float>(reduceExecuteSmemReduce<float>, "Shared Memory Reduce", 
                           d_input, d_output, N, (float)N);
    benchmark<float>(reduceExecuteShflReduce<float>, "Shuffle Reduce", 
                           d_input, d_output, N, (float)N);
    benchmark<float>(reduceExecuteShflReduceVec4<float>, "Shuffle Reduce Vec4", 
                           d_input, d_output, N, (float)N);

    // 4. 清理资源
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}