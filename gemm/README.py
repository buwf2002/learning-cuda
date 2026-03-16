


LHS : RHS both of Row-Major

Tips: block x,y,z这个顺序, 假如是x维度上先凑成一个warp, 在矩阵维度映射的时候, 变化快的维度, 对应x, 维度
1. 

int m_idx = blockIdx.x * BM + threadIdx.x;
int n_idx = blockIdx.y * BN + threadIdx.y;

| Baseline        |   (512, 512, 512) |   (8192, 8192, 8192) |
|-----------------|-------------------|----------------------|
| GeMM Naive      |           2473.98 |          9.41266e+06 |
| GeMM Block Tile |            993.28 |          3.99675e+06 |

2. 

int n_idx = blockIdx.x * BN + threadIdx.x;
int m_idx = blockIdx.y * BM + threadIdx.y;

| Baseline        |   (512, 512, 512) |   (8192, 8192, 8192) |
|-----------------|-------------------|----------------------|
| GeMM Naive      |            309.25 |          1.3939e+06  |
| GeMM Block Tile |            261.12 |          1.01921e+06 |
