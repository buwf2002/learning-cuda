## GEMM with SIMT (CUDA Core)

**GEMM Formula:** `C = A × B`, where A, B, C are matrices with dimensions `(M×K)`, `(K×N)`, and `(M×N)` respectively.


### Baseline

LHS × RHS, both Row-Major

| Baseline(unit:us)                    |   (512, 512, 512) |   (4096, 4096, 4096) |   (8192, 8192, 8192) |
|--------------------------------------|-------------------|----------------------|----------------------|
| Torch Matmul                         |             22.53 |              3586.96 |              28594.2 |
| Torch Compile                        |             22.5  |              3574.62 |              28338.2 |
| GeMM Naive                           |            120.86 |             43993.6  |             452375   |
| GeMM Block Tile                      |             88.06 |             31353.9  |             274077   |
| GeMM Block Tile Double buffer        |             83.97 |             29892.6  |             257117   |
| GeMM Block Thread Tile               |            126.98 |              7118.85 |              52018.2 |
| GeMM Block Thread Tile Double Buffer |            117.36 |              9826.3  |              77100.1 |

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
