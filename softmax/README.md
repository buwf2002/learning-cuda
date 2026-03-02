# Softmax CUDA 优化学习

记录学习 Softmax CUDA 优化的过程与心得。

## Baseline 实现

以下是不同优化级别的 Softmax 实现方案：

1. **torch.softmax(dim=-1)**  
   PyTorch 原生实现，作为性能基准。

2. **torch.compile(torch.softmax)**  
   `torch.compile` 使用即时编译（JIT）技术将 PyTorch 代码转换为优化的机器代码，会对所有参数进行特化优化。

3. **Softmax SMEM Block Reduce (SBR)**  
   一个 Thread Block 负责处理输入矩阵的一行，直接借助共享内存（Shared Memory）进行树状规约（Tree Reduction），计算 sum/max。

4. **Softmax SMEM Block Reduce using Vec 4 (SBR V4)**  
   在 SBR 的基础上引入向量化加载指令（`float4`），每次从全局内存读取 4 个 float 元素，提升内存带宽利用率。

5. **Softmax Warp Reduce using Vec 4 (WR V4)**  
   一个 Thread Block 负责一行，先进行 Warp 内部规约，再借助 Shared Memory 进行 Warp 之间的规约（即 Block Reduce）。

6. **Softmax SMEM Block Reduce using Vec 4 + Warp Reduce (SBR WR V4)**  
   一个 Thread Block 负责一行，先借助 Shared Memory 在 Block 内部进行树状规约。与之前方案不同的是，只规约到第一个 Warp，然后在 Warp 内部使用 `__shfl_sync` 指令进行规约，减少 Warp 分歧（Warp Divergence）。

## 实现结果

测试配置：`BLOCK_SIZE = 512`


| Baseline          |   (1024, 512) |   (8192, 8192) |   (1, 128256) |
|-------------------|---------------|----------------|---------------|
| PyTorch           |         20.48 |        2364.42 |         38.91 |
| torch.compile     |         44.03 |        4652.03 |         31.74 |
| Softmax SBR       |         30.72 |        2321.41 |         39.94 |
| Softmax SBR V4    |         25.6  |        2337.79 |         33.79 |
| Softmax WR V4     |         22.53 |        2337.79 |         33.79 |
| Softmax SBR WR V4 |         22.59 |        2335.74 |         33.79 |

## 踩坑记录

1. **Warp Reduce 循环内部不需要 if 分支**  
   Warp Reduce 循环内部如果使用 `if` 条件判断，会导致部分 Thread 不执行 `__shfl_sync` 指令，而其他 Thread 仍在等待，造成死锁卡死。
2. **Benchmark 函数内部不要调用 `cudaDeviceSynchronize()`**  
   在性能测试时，如果在每次 kernel 执行后都调用 `cudaDeviceSynchronize()`，会导致测量结果偏慢。原因如下：
   - CUDA kernel 是异步执行的，启动后 CPU 会立即返回继续执行后续代码，而 GPU 在后台并行计算。
   - `cudaDeviceSynchronize()` 会阻塞 CPU 直到 GPU 完成所有任务，频繁调用会破坏 CPU 和 GPU 的流水线并行。