import torch
from triton.testing import do_bench
import torch.utils.cpp_extension
from tabulate import tabulate

def benchmark(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3  # return in us

if __name__ == "__main__":
    my_module = torch.utils.cpp_extension.load(
        "module",
        sources=["gemm_simt.cu", "gemm.cpp"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        verbose=True,
    )
    compiled_matmul = torch.compile(torch.matmul)
    shapes = [
        (512, 512, 512),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    baselines = {
        "Torch Matmul": (
            lambda lhs, rhs: benchmark(torch.matmul, lhs, rhs),
            lambda lhs, rhs: torch.matmul(lhs, rhs)
        ),
        "Torch Compile": (
            lambda lhs, rhs: benchmark(compiled_matmul, lhs, rhs),
            lambda lhs, rhs: compiled_matmul(lhs, rhs)
        ),
        "GeMM Naive": (
            lambda lhs, rhs: benchmark(my_module.gemm_naive, lhs, rhs),
            lambda lhs, rhs: my_module.gemm_naive(lhs, rhs)
        ),
        "GeMM Block Tile": (
            lambda lhs, rhs: benchmark(my_module.gemm_block_tile, lhs, rhs),
            lambda lhs, rhs: my_module.gemm_block_tile(lhs, rhs)
        ),
        "GeMM Block Tile Double buffer": (
            lambda lhs, rhs: benchmark(my_module.gemm_block_tile_double_buffer, lhs, rhs),
            lambda lhs, rhs: my_module.gemm_block_tile_double_buffer(lhs, rhs)
        ),
        "GeMM Block Thread Tile": (
            lambda lhs, rhs: benchmark(my_module.gemm_block_thread_tile, lhs, rhs),
            lambda lhs, rhs: my_module.gemm_block_thread_tile(lhs, rhs)
        )
    }

    # Build results table: rows = baselines, columns = shapes
    table_data = []
    for name, benrch_fns in baselines.items():
        row = [name]
        for M, N, K in shapes:
            lhs = torch.randn(M, K).cuda()
            rhs = torch.randn(K, N).cuda()  # B is (N x K), row major (stored as B^T)
            # Our kernel computes C = A * B^T, so we need rhs.T for torch.matmul
            output_ref = torch.matmul(lhs, rhs)
            output = benrch_fns[1](lhs, rhs)
            # use_fast_math causes some numerical precision loss
            torch.testing.assert_close(output, output_ref, atol=1e-3, rtol=1e-3, msg=f"output mismatch for {name} at shape ({M}, {N}, {K})")
            time_us = benrch_fns[0](lhs, rhs)
            row.append(f"{time_us:.2f}")
        table_data.append(row)

    # Create header
    header = ["Baseline(unit:us)"] + [f"({M}, {N}, {K})" for M, N, K in shapes]

    # Print as markdown table
    print()
    print(tabulate(table_data, headers=header, tablefmt="github"))