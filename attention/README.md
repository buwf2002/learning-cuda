# FlashAttention 学习笔记

本文记录学习 FlashAttention 的原理与CUDA实现。

- 学习资料源自
**特别注意Online softmax - Implementation**
[FlashAttention Blog](https://gau-nernst.github.io/fa-5090/)

我在这个博客的基础上做了两点改动:
- 增加 GQA 和 causal mask。
- 修改写回方式。
  并通过 Nsight Compute 观察到写回没有合并访存，写回效率只有50%，所以通过 shared memory 优化了写回，使得写回效率100%。但这一过程因为引入了额外的 shared memory 写回操作，性能并没有提升(有空再看看能不能继续优化)。


---

## 目录

1. [Attention 背景介绍](#1-attention-背景介绍)
2. [MHA 与 GQA](#2-mha-与-gqa)
3. [FlashAttention 原理](#3-flashattention-原理)
4. [Online Softmax](#4-online-softmax)
5. [V3 代码解析](#5-v3-代码解析)
6. [各版本实验结果对比](#6-各版本实验结果对比)

---

## 1. Attention 背景介绍

### 1.1 Self-Attention 公式

Transformer 中的核心操作 Self-Attention 定义为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$: Query 矩阵
- $K \in \mathbb{R}^{m \times d_k}$: Key 矩阵
- $V \in \mathbb{R}^{m \times d_v}$: Value 矩阵
- $d_k$: 缩放因子，防止点积过大

### 1.2 传统 Attention 的问题

**标准实现流程：**
```python
# PyTorch 伪代码
S = Q @ K.transpose(-2, -1)  # [n, m] - 注意力分数
S = S / sqrt(d_k)            # 缩放
if causal:
    mask = torch.triu(torch.ones(n, m), diagonal=1) * -inf
    S = S.masked_fill(mask, -inf)
P = softmax(S, dim=-1)       # [n, m] - 注意力权重
O = P @ V                    # [n, d_v] - 输出
```
---

## 2. MHA 与 GQA

### 2.1 Multi-Head Attention (MHA)

多头注意力将查询/键/值投影到多个子空间并行计算：

```
Q: [bs, q_head, len_q, dim]
K: [bs, q_head, len_kv, dim]   # q_head == kv_head
V: [bs, q_head, len_kv, dim]
```

**特点：**
- 每个 head 独立计算 attention
- 表达能力强，但 KV Cache 显存占用大

### 2.2 Grouped-Query Attention (GQA)

GQA 是 MHA 和 MQA (Multi-Query Attention) 的折中方案：

```
Q: [bs, q_head, len_q, dim]
K: [bs, kv_head, len_kv, dim]  # kv_head < q_head
V: [bs, kv_head, len_kv, dim]

q_group = q_head / kv_head  # 每组 query 共享一组 KV
```

**示例配置：**

| 模型 | q_head | kv_head | q_group | 类型 |
|------|--------|---------|---------|------|
| LLaMA-7B | 32 | 32 | 1 | MHA |
| LLaMA-2-70B | 64 | 8 | 8 | GQA |
| LLaMA-3-70B | 64 | 8 | 8 | GQA |

**优势：**
- KV Cache 显存减少 `q_group` 倍
- 推理吞吐量提升，质量损失小

**代码中的 GQA 处理：**
```cpp
int q_group = q_head / kv_head;

// Q 按 (bs_id * num_q_blocks + q_block_id) 索引
Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;

// K, V 按 (bs_id / q_group) 索引，实现 KV 共享
K += (bs_id / q_group) * len_kv * DIM;
V += (bs_id / q_group) * len_kv * DIM;
```

---

## 3. FlashAttention 原理

### 3.1 核心思想

**分块计算 (Tiling) + 重计算 (Recomputation)**

将 $n \times m$ 的大矩阵分块为 $B_r \times B_c$ 的小块，在 SRAM 中完成计算：

```

```

### 3.2 算法流程

**FlashAttention v1 算法：**

```
Algorithm 1: FlashAttention
Input: Q (N×d), K (M×d), V (M×d)
Output: O (N×d)

1: Partition Q into Tr tiles of size Br×d
2: Partition K, V into Tc tiles of size Bc×d
3: Initialize O, ℓ, m (output, rowsum, rowmax)
4: Each thread block i load Qi from HBM to SRAM.
5: for j = 1 to Tc do                    // 外层循环：KV 块
6:   Load Kj, Vj from HBM to SRAM
7:     Compute Sij = Qi @ Kj^T
8:     Update rowmax m_i and rescale O_i
9:    Compute Pij = exp(Sij - m_i)
10:    Update rowsum ℓ_i
11:    Compute O_i += Pij @ Vj
12:    Write O_i back to HBM
```

---

## 4. Online Softmax

### 4.1 问题：分块 Softmax 的不一致性

Softmax 需要全局最大值进行归一化：

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

分块计算时，每个块的局部最大值不同，如何保证结果正确？

### 4.2 Online Softmax 原理

**核心观察：**
$$\text{softmax}(x) = \frac{e^{x - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

减去最大值不影响结果，但提供数值稳定性。

**增量更新公式：**

假设已有前 $k$ 个元素的统计量 $(m_k, \ell_k, o_k)$：
- $m_k$: 前 $k$ 个元素的最大值
- $\ell_k$: 前 $k$ 个元素的归一化分母（带缩放）
- $o_k$: 前 $k$ 个元素的加权和（带缩放）

加入第 $k+1$ 块后：

```
m_new = max(m_old, m_block)           # 新的全局最大值
α     = exp(m_old - m_new)            # 旧输出的缩放因子
β     = exp(m_block - m_new)          # 新块的缩放因子
ℓ_new = α * ℓ_old + β * ℓ_block       # 更新分母
o_new = α * o_old + β * o_block       # 更新分子
```

### 4.3 代码实现

```cpp
// 初始化
float rowmax[WARP_Q / MMA_M][2];
float rowsumexp[WARP_Q / MMA_M][2] = {};

for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
  rowmax[mma_id_q][0] = -FLT_MAX;
  rowmax[mma_id_q][1] = -FLT_MAX;
}

// 每个 KV 块的更新循环
for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
  // ... 计算 S = Q @ K.T ...

  // 1. 计算当前块的局部 rowmax
  float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
  for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
    float *regs = S_rmem[mma_id_q][mma_id_kv];
    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
  }
  // Warp 内归约 (Butterfly Reduction)
  this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
  this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));

  // 2. 更新全局 rowmax
  float prev_max0 = rowmax[mma_id_q][0];
  float prev_max1 = rowmax[mma_id_q][1];
  rowmax[mma_id_q][0] = max(prev_max0, this_rowmax[0]);
  rowmax[mma_id_q][1] = max(prev_max1, this_rowmax[1]);

  // 3. 计算缩放因子并 rescale 之前的 O
  float rescale[2];
  rescale[0] = __expf(prev_max0 - rowmax[mma_id_q][0]);
  rescale[1] = __expf(prev_max1 - rowmax[mma_id_q][1]);
  for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
    O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
    O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
    O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
    O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
  }

  // 4. 计算当前块的 exp 并累加到 rowsumexp
  float this_rowsumexp[2] = {};
  for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
    float *regs = S_rmem[mma_id_q][mma_id_kv];
    regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
    regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
    regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
    regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

    this_rowsumexp[0] += regs[0] + regs[1];
    this_rowsumexp[1] += regs[2] + regs[3];
  }
  // Warp 内归约
  this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
  this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);

  // 5. 更新全局 rowsumexp (带缩放)
  rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
  rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];

  // 6. 计算 O += P @ V
  // ...
}

// 最后归一化
for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
  for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
    float *regs = O_rmem[mma_id_q][mma_id_d];
    regs[0] /= rowsumexp[mma_id_q][0];
    regs[1] /= rowsumexp[mma_id_q][0];
    regs[2] /= rowsumexp[mma_id_q][1];
    regs[3] /= rowsumexp[mma_id_q][1];
  }
```

### 4.4 Butterfly Reduction 图解

```
4 线程归约 (lane_id: 0,1,2,3):

Step 1: __shfl_xor(x, 1)  - 相邻交换
  lane0 ← max(lane0, lane1)
  lane1 ← max(lane0, lane1)
  lane2 ← max(lane2, lane3)
  lane3 ← max(lane2, lane3)

Step 2: __shfl_xor(x, 2)  - 跨步交换
  lane0 ← max(lane0, lane2)  ← 最终结果
  lane1 ← max(lane1, lane3)
  lane2 ← max(lane0, lane2)
  lane3 ← max(lane1, lane3)

结果：所有线程都有全局最大值
```

---

## 5. V3 代码解析

### 5.1 V3 的核心优化

相比 V1/V2，V3 引入了 **异步内存拷贝 (Async Copy) + 双缓冲 (Double Buffering)**：

| 版本 | 关键特性 |
|------|---------|
| V1 | 基础实现，同步加载 |
| V2 | 添加 Swizzle 优化 |
| **V3** | **异步拷贝 + 双缓冲预取** |
| V4 | ldmatrix x4 优化 |
| V5 | 分离 K/V 缓冲策略 |
| V6 | 输出 Swizzle + 优化存储 |

### 5.2 内核启动配置

```cpp
void attention_v3(...) {
  const int BLOCK_Q = 64;      // Q 块大小
  const int BLOCK_KV = 32;     // KV 块大小 (V3 减小到 32)
  const int DIM = 128;         // 隐藏维度
  const int NUM_WARPS = 4;     // Warp 数量

  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;  // 128 线程

  // 共享内存：Q 与 (K, V) 复用，K 使用双缓冲
  const int smem_size = max(BLOCK_Q, BLOCK_KV * 2 * 2) * DIM * sizeof(nv_bfloat16);
  //            Q: 64 * 128 * 2 = 16KB
  //            K: 32 * 2 * 128 * 2 = 16KB (双缓冲)
  //            V: 同上，与 K 重叠布局
}
```

### 5.3 共享内存布局

```cpp
extern __shared__ nv_bfloat16 smem[];
const uint32_t Q_smem = __cvta_generic_to_shared(smem);
const uint32_t K_smem = Q_smem;  // 与 Q 重叠（Q 只需加载一次）
const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);
```

**内存复用策略：**
```
时间线：
1. 加载 Q → Q_smem → 加载到寄存器 → Q_smem 可复用
2. 循环加载 K/V:
   iter0: K_smem[buf0], V_smem[buf0]
   iter1: K_smem[buf1], V_smem[buf1]  (双缓冲)
   iter0: K_smem[buf0], V_smem[buf0]  (覆盖)
```

### 5.4 异步拷贝与双缓冲

**V3 的关键改进：使用 `cp.async` 进行异步预取**

```cpp
// 定义加载函数（Lambda）
auto load_K = [&](int kv_id) {
  if (kv_id < num_kv_iter) {
    // 双缓冲：kv_id % 2 选择缓冲区
    const uint32_t dst = K_smem + (kv_id % 2) * (2 * BLOCK_KV * DIM * sizeof(nv_bfloat16));
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, K, DIM, tid);
    K += BLOCK_KV * DIM;
  }
  asm volatile("cp.async.commit_group;");  // 提交拷贝
};

auto load_V = [&](int kv_id) {
  if (kv_id < num_kv_iter) {
    const uint32_t dst = V_smem + (kv_id % 2) * (2 * BLOCK_KV * DIM * sizeof(nv_bfloat16));
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, V, DIM, tid);
    V += BLOCK_KV * DIM;
  }
  asm volatile("cp.async.commit_group;");
};

// 主循环：预取下一块
load_K(0);  // 预取第一块
load_V(0);

for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
  // 1. 预取下一个 K 块（与计算重叠）
  load_K(kv_id + 1);

  // 2. 等待当前 K 块就绪
  asm volatile("cp.async.wait_group 2;");
  __syncthreads();

  // 3. K: shared → registers (计算 S = Q @ K.T)
  // ...

  // 4. 预取下一个 V 块
  load_V(kv_id + 1);

  // 5. 计算 softmax 和 P @ V
  // ...

  // 6. 等待 V 块就绪
  asm volatile("cp.async.wait_group 2;");
  __syncthreads();

  // 7. V: shared → registers (计算 O += P @ V)
  // ...
}
```

**流水线示意图：**
```
时间 →
V1/V2 (同步):  [Load K] → [Compute S] → [Load V] → [Compute O]
V3 (异步):     [Load K] → [Compute S] → [Load V] → [Compute O]
               └─ 预取 ─┘   └─ 预取 ─┘
               计算与拷贝重叠！
```

### 5.5 Swizzle 优化

**问题：** Shared Memory 的 Bank Conflict

```cpp
// V2 引入的 Swizzle 预计算
uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
{
  // A tile (Q 矩阵)
  const int row_off = warp_id * WARP_Q + (lane_id % 16);
  const int col_off = lane_id / 16 * 8;
  Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
}
```

**Swizzle 原理：**
```
原始地址：addr = row * stride + col
Swizzle 后：addr' = addr XOR ((row % 8) >> shift) << 4

效果：将连续的行地址分散到不同的 Memory Bank
避免 32 线程同时访问同一 Bank 造成的串行化
```

### 5.6 Tensor Core MMA 操作

**MMA 矩阵乘法 (m16n8k16)：**

```cpp
// 寄存器布局
uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];   // 4 个寄存器/线程
uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2]; // 2 个寄存器/线程
float    O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4];   // 累加器

// MMA 计算 S = Q @ K.T
for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
  for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
      mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],
                   K_rmem[mma_id_kv][mma_id_d],
                   S_rmem[mma_id_q][mma_id_kv]);
```

**MMA 指令格式：**
```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
  {D0, D1, D2, D3},    // 2x2 float 结果 (每个线程)
  {A0, A1, A2, A3},    // 16x16 bf16 矩阵 A
  {B0, B1},            // 16x8 bf16 矩阵 B
  {C0, C1, C2, C3}     // 累加器
```

### 5.7 Causal Mask 实现

```cpp
// 在寄存器级别应用因果掩码
for (int reg_id = 0; reg_id < 4; reg_id++) {
  S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

  if (causal) {
    // 计算全局位置
    int global_q = (reg_id < 2) ? global_q_pos0 : global_q_pos1;
    int global_k = (reg_id % 2 == 0) ? global_k_pos0 : global_k_pos1;

    // 掩码：k_pos > q_pos (未来位置)
    if (global_k > global_q)
      S_rmem[mma_id_q][mma_id_kv][reg_id] = -FLT_MAX;
  }
}
```
---

## 6. 各版本实验结果对比
### 6.1 版本演进总结

实验GPU: NVIDIA GeForce RTX 4060 Laptop GPU

| Kernel           |   Latency (ms) |   TFLOPS |
|:-----------------|---------------:|---------:|
| F.sdpa() - FA    |         5.2808 |    52.05 |
| F.sdpa() - CuDNN |         4.8333 |    56.87 |
| flash-attn       |         4.8302 |    56.91 |
| v1               |        14.0902 |    19.51 |
| v2               |         5.1999 |    52.86 |
| v3               |         5.1292 |    53.59 |
| v4               |         4.8701 |    56.44 |
| v5               |         4.9562 |    55.46 |
| v6               |         4.9756 |    55.25 |

### 6.3 版本信息

```
V1: 基于Tensor Core的最基础实现

V1 → V2: Swizzle 解决 Bank Conflict

V2 → V3: 异步拷贝隐藏内存延迟

V3 → V4: ldmatrix x4 减少指令数

V4 → V5: V 单缓冲减少

V5 → V6: 输出 Swizzle + 优化存储
```

