import torch
from triton.testing import do_bench
import torch.utils.cpp_extension
from tabulate import tabulate

def benchmark(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3  # return in us

if __name__ == "__main__":
    my_module = torch.utils.cpp_extension.load(
        "module",
        sources=["softmax.cu", "softmax.cpp"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=True,
    )
    compiled_softmax = torch.compile(torch.softmax, mode="max-autotune", dynamic=False)

    shapes = [
        (1024, 512),
        (8192, 8192),
        (1, 128256),
    ]

    baselines = {
        "PyTorch": (
            lambda inp: benchmark(torch.softmax, inp, 1),
            lambda inp: torch.softmax(inp, dim=1) # 增加 dim=1
        ),
        "torch.compile": (
            lambda inp: benchmark(compiled_softmax, inp, 1),
            lambda inp: compiled_softmax(inp, dim=1) # 增加 dim=1
        ),
        "Softmax SBR": (
            lambda inp: benchmark(my_module.softmax_sbr, inp),
            my_module.softmax_sbr
        ),
        "Softmax SBR V4": (
            lambda inp: benchmark(my_module.softmax_sbr_v4, inp),
            my_module.softmax_sbr_v4
        ),
        "Softmax WR V4": (
            lambda inp: benchmark(my_module.softmax_wr_v4, inp),
            my_module.softmax_wr_v4
        ),
        "Softmax SBR WR V4": (
            lambda inp: benchmark(my_module.softmax_sbr_wr_v4, inp),
            my_module.softmax_sbr_wr_v4
        ),
    }

    # Build results table: rows = baselines, columns = shapes
    table_data = []
    for name, benrch_fns in baselines.items():
        row = [name]
        for M, N in shapes:
            input = torch.randn(M, N).cuda()
            output_ref = torch.softmax(input, dim=1)
            output = benrch_fns[1](input)
            assert torch.allclose(output, output_ref, atol=1e-5), f"Output mismatch for {name} at shape ({M}, {N})"            
            time_us = benrch_fns[0](input)
            row.append(f"{time_us:.2f}")
        table_data.append(row)

    # Create header
    header = ["Baseline"] + [f"({M}, {N})" for M, N in shapes]

    # Print as markdown table
    print(tabulate(table_data, headers=header, tablefmt="github"))