import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from triton.testing import do_bench

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

CURRENT_DIR = Path(__file__).parent

module = load(
    "my_ext",
    sources=list(CURRENT_DIR.glob("attention*")),
    extra_cuda_cflags=["-lineinfo", "--ptxas-options=-v"],
    verbose=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--nh_q", type=int, default=8)
    parser.add_argument("--nh_kv", type=int, default=4)
    parser.add_argument("--lq", type=int, default=4096)
    parser.add_argument("--lkv", type=int, default=4096)
    parser.add_argument("--causal", type=bool, default=True)
    args = parser.parse_args()

    bs = args.bs
    nh_q = args.nh_q
    nh_kv = args.nh_kv
    lq = args.lq
    lkv = args.lkv
    head_dim = 128
    causal = args.causal

    # add a small offset so that output does not have a mean of zero,
    # which will result in large relative error
    def generate_input(*shape):
        return torch.randn(shape).add(0.5).bfloat16().cuda()

    Q = generate_input(bs, nh_q, lq, head_dim)
    K = generate_input(bs, nh_kv, lkv, head_dim)
    V = generate_input(bs, nh_kv, lkv, head_dim)

    if args.profile is not None:
        if args.profile == "fa":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V, is_causal=causal, enable_gqa=True)

        elif args.profile == "cudnn":
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V, is_causal=causal, enable_gqa=True)

        else:
            f = getattr(module, f"sdpa_v{args.profile}")
            f(Q, K, V, causal)

        torch.cuda.synchronize()
        return

    # https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
    # Peak FP16 Tensor TFLOPSwith FP32 Accumulate
    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
        "NVIDIA GeForce RTX 4060 Laptop GPU": 29.1,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 0)

    results = []

    def bench_and_print(f, name, **args):
        # sleep to stabilize thermal
        time.sleep(1)

        latency_ms = do_bench(lambda: f(**args), return_mode="median")
        tflops = 4 * bs * nh_q * lq * lkv * head_dim / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        results.append([name, round(latency_ms, 4), round(tflops, 2), round(pct_sol, 2)])

    out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, enable_gqa=True)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - FA", query=Q, key=K, value=V, is_causal=causal, enable_gqa=True)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - CuDNN", query=Q, key=K, value=V, is_causal=causal, enable_gqa=True)

    if flash_attn_func is not None:
        out = flash_attn_func(q=Q.transpose(1, 2), k=K.transpose(1, 2), v=V.transpose(1, 2), causal=causal).transpose(1, 2)
        torch.testing.assert_close(out, out_ref)
        bench_and_print(
            flash_attn_func,
            "flash-attn",
            q=Q.transpose(1, 2),
            k=K.transpose(1, 2),
            v=V.transpose(1, 2),
            causal=causal
        )

    for i in range(6):
        f = getattr(module, f"sdpa_v{i + 1}")
        out = f(Q, K, V, causal)
        torch.testing.assert_close(out, out_ref, atol=1.6e-2, rtol=1e-4)
        bench_and_print(f, f"v{i + 1}", Q=Q, K=K, V=V, causal=causal)

    df = pd.DataFrame(results, columns=["Kernel", "Latency (ms)", "TFLOPS", "% SOL"])
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
