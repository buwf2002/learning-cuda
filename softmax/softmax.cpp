#include <torch/extension.h>

using softmax_fn = void (*)(const float *input, float *output, int M, int N);

// Forward declaration of the CUDA kernel launcher
void softmaxNaiveExec(const float* input, float* output, int m, int n);
void softmaxVec4Exec(const float* input, float* output, int m, int n);
void softmaxVec4SmemTreeExec(const float* input, float* output, int m, int n);

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
    m.def("softmax_naive", &softmax<softmaxNaiveExec>, "Naive softmax", py::arg("input"));
    m.def("softmax_vec4", &softmax<softmaxVec4Exec>, "Naive softmax", py::arg("input"));
    m.def("softmax_vec4_smem_tree", &softmax<softmaxVec4SmemTreeExec>, "Naive softmax", py::arg("input"));
}
