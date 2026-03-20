## GEMM with SIMT (CUDA Core)

### Introduction

General Matrix Multiplication (GEMM) is a fundamental operation in deep learning and high-performance computing. This module explores GEMM optimization using CUDA's SIMT (Single Instruction, Multiple Threads) architecture on CUDA Cores.

**GEMM Formula:** `C = A × B`, where A, B, C are matrices with dimensions `(M×K)`, `(K×N)`, and `(M×N)` respectively.

**Key Optimization Techniques:**
- Thread block tiling for shared memory reuse
- Warp-level parallelism with proper thread indexing
- Memory access pattern optimization (row-major layout)

### Baseline

LHS × RHS, both Row-Major

| Baseline                      |   (512, 512, 512) |   (4096, 4096, 4096) |   (8192, 8192, 8192) |
|-------------------------------|-------------------|----------------------|----------------------|
| Torch Matmul                  |             58.37 |              15697.9 |     130901           |
| Torch Compile                 |             56.32 |              15795.2 |     130235           |
| GeMM Naive                    |            310.18 |             171854   |          1.38876e+06 |
| GeMM Block Tile               |            261.12 |             125618   |          1.02578e+06 |
| GeMM Block Tile Double buffer |            254.98 |             119345   |     950327           |

> **Tip:** Thread indexing order matters. The `x` dimension forms warps first. When mapping to matrix dimensions, the faster-changing dimension should correspond to the `x` dimension.

#### Configuration 1: `threadIdx.x → M`, `threadIdx.y → N`

```cpp
int m_idx = blockIdx.x * BM + threadIdx.x;
int n_idx = blockIdx.y * BN + threadIdx.y;
```

| Baseline        |   (512, 512, 512) |   (8192, 8192, 8192) |
|-----------------|-------------------|----------------------|
| GeMM Naive      |           2473.98 |          9.41266e+06 |
| GeMM Block Tile |            993.28 |          3.99675e+06 |

#### Configuration 2: `threadIdx.x → N`, `threadIdx.y → M`

```cpp
int n_idx = blockIdx.x * BN + threadIdx.x;
int m_idx = blockIdx.y * BM + threadIdx.y;
```

| Baseline        |   (512, 512, 512) |   (8192, 8192, 8192) |
|-----------------|-------------------|----------------------|
| GeMM Naive      |            309.25 |          1.3939e+06  |
| GeMM Block Tile |            261.12 |          1.01921e+06 |
