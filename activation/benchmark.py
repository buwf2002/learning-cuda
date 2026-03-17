import torch
from triton.testing import do_bench
import torch.utils.cpp_extension
from tabulate import tabulate

def benchmark(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3  # return in us

if __name__ == "__main__":
    my_module = torch.utils.cpp_extension.load(
        "module",
        sources=["activation.cu", "activation.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=True,
    )

    shapes = [
        (1024, 512),
        (8192, 8192),
        (1, 128256),
    ]

    # Sigmoid benchmarks
    print("\nSigmoid:")
    sigmoid_baselines = {
        "PyTorch": (
            lambda inp: benchmark(torch.sigmoid, inp),
            torch.sigmoid
        ),
        "Sigmoid Naive": (
            lambda inp: benchmark(my_module.sigmoid_naive, inp),
            my_module.sigmoid_naive
        ),
        "Sigmoid Vec4": (
            lambda inp: benchmark(my_module.sigmoid_vec4, inp),
            my_module.sigmoid_vec4
        ),
    }
    
    table_data = []
    for name, bench_fns in sigmoid_baselines.items():
        row = [name]
        for M, N in shapes:
            input_tensor = torch.randn(M, N).cuda()
            output_ref = torch.sigmoid(input_tensor)
            output = bench_fns[1](input_tensor)
            assert torch.allclose(output, output_ref, atol=1e-5), f"Output mismatch for {name} at shape ({M}, {N})"
            time_us = bench_fns[0](input_tensor)
            row.append(f"{time_us:.2f}")
        table_data.append(row)
    
    header = ["Baseline"] + [f"({M}, {N})" for M, N in shapes]
    print(tabulate(table_data, headers=header, tablefmt="github"))

    # ReLU benchmarks
    print("\nReLU:")
    relu_baselines = {
        "PyTorch": (
            lambda inp: benchmark(torch.relu, inp),
            torch.relu
        ),
        "ReLU Naive": (
            lambda inp: benchmark(my_module.relu_naive, inp),
            my_module.relu_naive
        ),
        "ReLU Vec4": (
            lambda inp: benchmark(my_module.relu_vec4, inp),
            my_module.relu_vec4
        ),
    }
    
    table_data = []
    for name, bench_fns in relu_baselines.items():
        row = [name]
        for M, N in shapes:
            input_tensor = torch.randn(M, N).cuda()
            output_ref = torch.relu(input_tensor)
            output = bench_fns[1](input_tensor)
            assert torch.allclose(output, output_ref, atol=1e-5), f"Output mismatch for {name} at shape ({M}, {N})"
            time_us = bench_fns[0](input_tensor)
            row.append(f"{time_us:.2f}")
        table_data.append(row)
    
    print(tabulate(table_data, headers=header, tablefmt="github"))

    # LayerNorm benchmarks
    print("\nLayerNorm:")
    ln_baselines = {
        "PyTorch": (
            lambda inp: benchmark(torch.nn.functional.layer_norm, inp, [N]),
            lambda inp: torch.nn.functional.layer_norm(inp, [N])
        ),
        "LayerNorm Naive": (
            lambda inp: benchmark(my_module.layernorm_naive, inp, N, 1e-5),
            lambda inp: my_module.layernorm_naive(inp, N, 1e-5)
        ),
        "LayerNorm Vec4": (
            lambda inp: benchmark(my_module.layernorm_vec4, inp, N, 1e-5),
            lambda inp: my_module.layernorm_vec4(inp, N, 1e-5)
        ),
    }
    
    table_data = []
    for name, bench_fns in ln_baselines.items():
        row = [name]
        for M, N in shapes:
            input_tensor = torch.randn(M, N).cuda()
            output_ref = torch.nn.functional.layer_norm(input_tensor, [N])
            output = bench_fns[1](input_tensor)
            assert torch.allclose(output, output_ref, atol=1e-5), f"Output mismatch for {name} at shape ({M}, {N})"
            time_us = bench_fns[0](input_tensor)
            row.append(f"{time_us:.2f}")
        table_data.append(row)
    
    print(tabulate(table_data, headers=header, tablefmt="github"))
