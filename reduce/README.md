## Reduce 优化记录
- N = 1000000
---
### naive

直接每个线程读取对应元素，然后atomicAdd

- grid-stride loop 和直接开够足够的线程好像性能没有太大区别.

```shell
# naive(enough blocks)
--------------------------------------
Kernel:                     Naive (Global AtomicAdd)
Average Time per iteration: 1.41816 ms
Effective Bandwidth:        2.95577 GB/s
--------------------------------------

# grid-stride loop
--------------------------------------
Kernel:                     Naive (Global AtomicAdd)
Average Time per iteration: 1.43167 ms
Effective Bandwidth:        2.9433 GB/s
--------------------------------------
```
---
### warp-level reduce

Naive Kernel 存在的问题: atomicAdd 的争用 使得thread之间完全的串行

warp-level reduce 让每个warp内部只有一个线程进行累加，然后再warp之间进行atomicAdd

这样可以显著的降低atomicAdd 的争用, 利用warp 之间的并行增加并行度.

```shell
Average Time per iteration: 0.0489635 ms
Effective Bandwidth:        85.446 GB/s
```
---
### block-level reduce

block-level reduce 让每个 block 内部只有一个线程进行累加，然后再block之间进行atomicAdd

这样可以进一步的降低 atomicAdd 的争用, 但是相比warp-level 因为每个block只有一个线程在工作，所以相比 warp-level reduce 并行度下降，性能下降

```shell
Average Time per iteration: 0.069161 ms
Effective Bandwidth:        85.446 GB/s
```
---
### block-level reduce using smem.

smem 树状规约

block-level reduce Kernel 存在的问题: 虽然 atomicAdd 最低，但是并行度很低.

block-level reduce using smem 让每个 block 内部所有的线程利用 smem 进行累加:
1. thread 将所有数据 load 到 smem
2. 每次用一半 thread 去 累加, 随着需要累加的有效元素对半减少，有效的线程也直接对半减少，知道最后只有 thread 0 存储最后的结果
3. thread 0 atomicAdd 写回

这样可以保持 block-level 的低 atomicAdd, 又可以利用 shared memory 组织 block 大半的 thread 参与运算，提高并行度.

```shell
Average Time per iteration: 0.0390758 ms
Effective Bandwidth:        118.659 GB/s
```
---
### block-level reduce using smem and shfl.

lock-level reduce using smem Kernel 存在的问题: 越到后面，参与运算的 thread 越少，活跃的 warp 数减少，不利于指令级并行.

block-level reduce using smem and shfl 让每个 block 内部的每个 warp 先进行reduce，再让 warp 之间用 smem 进行 reduce:
1. thread 将所有数据 load 到 register
2. warp 内部 利用 shfl 进行 reduce
3. warp lane id == 0 thread 将 warp 的结果写入 smem ，方便后续收集 不同 warp 的结果
4. 第一个 warp 读取 smem，此时 不同 warp 的结果，全部汇总到 warp 0, 然后使用 shfl 进行 warp 内部的规约，最终结果汇总到 thread 0

这样可以保持 block-level 的低 atomicAdd, 能够保持每个 warp 在这个过程中始终都有 thread 活跃，保证 活跃的 warp 数，可以更好的隐藏延迟.

```shell
Average Time per iteration: 0.0201082 ms
Effective Bandwidth:        211.4 GB/s
```

### block-level reduce using smem and shfl(Vec4).

在上面版本的基础上，引入LDG.128, 向量化访问。

结果发现在N很大的时候效果很好, 更好的利用带宽。


```bash
N = 2 * 1024 * 1024
-------------------------------------
Kernel:                     Shuffle Reduce
Average Time per iteration: 0.0442285 ms
Effective Bandwidth:        189.665 GB/s
Expected Result:            2.09715e+06
Actual Result:              2.09715e+06
Status:                     PASSED ✅
--------------------------------------

--------------------------------------
Kernel:                     Shuffle Reduce Vec4
Average Time per iteration: 0.0333107 ms
Effective Bandwidth:        251.829 GB/s
Expected Result:            2.09715e+06
Actual Result:              2.09715e+06
Status:                     PASSED ✅
--------------------------------------
```

