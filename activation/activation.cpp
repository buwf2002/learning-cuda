#include <torch/extension.h>

using unary_fn = void (*)(const float *input, float *output, int numel);
using binary_fn = void (*)(const float *input, float *output, int M, int N);
using layernorm_fn = void (*)(const float *input, float *output, int M, int N, float eps);

// Forward declaration of CUDA kernel launchers
// Sigmoid
void sigmoid_exec_naive(const float* input, float* output, int numel);
void sigmoid_exec_vec4(const float* input, float* output, int numel);

// ReLU
void relu_exec_naive(const float* input, float* output, int numel);
void relu_exec_vec4(const float* input, float* output, int numel);

// LayerNorm
void layernorm_exec_naive(const float* input, float* output, int M, int N, float eps);
void layernorm_exec_vec4(const float* input, float* output, int M, int N, float eps);

// Unary function wrappers (element-wise, no shape dependency)
template<unary_fn fn>
torch::Tensor unary_op(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    int numel = input.numel();
    torch::Tensor output = torch::empty_like(input);
    fn(input.data_ptr<float>(), output.data_ptr<float>(), numel);
    return output;
}

// Binary function wrappers (M x N shape aware)
template<binary_fn fn>
torch::Tensor binary_op(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    if (input.dim() == 1) input = input.unsqueeze(0);
    int M = input.size(0);
    int N = input.size(1);
    torch::Tensor output = torch::empty_like(input);
    fn(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

// LayerNorm wrapper with eps parameter
template<layernorm_fn fn>
torch::Tensor layernorm_op(torch::Tensor input, int N, float eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    if (input.dim() == 1) input = input.unsqueeze(0);
    int M = input.size(0);
    torch::Tensor output = torch::empty_like(input);
    fn(input.data_ptr<float>(), output.data_ptr<float>(), M, N, eps);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Sigmoid
    m.def("sigmoid_naive", &unary_op<sigmoid_exec_naive>, "Naive sigmoid implementation");
    m.def("sigmoid_vec4", &unary_op<sigmoid_exec_vec4>, "Vec4 optimized sigmoid");

    // ReLU
    m.def("relu_naive", &unary_op<relu_exec_naive>, "Naive ReLU implementation");
    m.def("relu_vec4", &unary_op<relu_exec_vec4>, "Vec4 optimized ReLU");

    // LayerNorm
    m.def("layernorm_naive", &layernorm_op<layernorm_exec_naive>, "Naive LayerNorm implementation",
          py::arg("input"), py::arg("N"), py::arg("eps") = 1e-5);
    m.def("layernorm_vec4", &layernorm_op<layernorm_exec_vec4>, "Vec4 optimized LayerNorm",
          py::arg("input"), py::arg("N"), py::arg("eps") = 1e-5);
}
