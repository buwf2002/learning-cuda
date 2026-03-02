#include <torch/extension.h>

using softmax_fn = void (*)(const float *input, float *output, int M, int N);

// Forward declaration of the CUDA kernel launcher
void softmax_exec_sbr(const float* input, float* output, int m, int n);
void softmax_exec_sbr_v4(const float* input, float* output, int m, int n);
void softmax_exec_wr_v4(const float* input, float* output, int m, int n);
void softmax_exec_sbr_wr_v4(const float* input, float* output, int m, int n);

template<softmax_fn fn>
torch::Tensor softmax(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    if (input.dim() == 1) input = input.unsqueeze(0);
    int M = input.size(0);
    int N = input.size(1);
    torch::Tensor output = torch::empty_like(input);
    fn(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_sbr", &softmax<softmax_exec_sbr>, "Naive softmax", py::arg("input"));
    m.def("softmax_sbr_v4", &softmax<softmax_exec_sbr_v4>, "Naive softmax", py::arg("input"));
    m.def("softmax_wr_v4", &softmax<softmax_exec_wr_v4>, "Naive softmax", py::arg("input"));
    m.def("softmax_sbr_wr_v4", &softmax<softmax_exec_sbr_wr_v4>, "Naive softmax", py::arg("input"));
}
