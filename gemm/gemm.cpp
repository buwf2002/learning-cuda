#include <torch/extension.h>

using gemm_fn = void (*)(
    const float* A,
    const float* B,
    float *C,
    int m, int n, int k
);

// Forward declaration of the CUDA kernel launcher
void gemm_naive(const float* A, const float* B, float *C, int m, int n, int k);
void gemm_block_tile(const float* A, const float* B, float *C, int m, int n, int k);
void gemm_block_tile_double_buffer(const float* A, const float* B, float *C, int m, int n, int k);
void gemm_block_thread_tile(const float* A, const float* B, float *C, int m, int n, int k);

template<gemm_fn fn>
torch::Tensor gemm(
    torch::Tensor A,
    torch::Tensor B
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    torch::Tensor output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    fn(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm<gemm_naive>, "Naive gemm");
    m.def("gemm_block_tile", &gemm<gemm_block_tile>, "Block tile gemm");
    m.def("gemm_block_tile_double_buffer", &gemm<gemm_block_tile_double_buffer>, "Block tile gemm with double buffer");
    m.def("gemm_block_thread_tile", &gemm<gemm_block_thread_tile>, "Block and thread tile gemm without double buffer");
}
