"""Adaptive-threshold k-WTA CUDA operator (Krotov & Hopfield style).

Unlike exact top-k (which sorts), the biological winnow uses per-neuron
adaptive thresholds iterated until ~k winners pass. Pure O(d) streaming,
no sort. Built as a torch extension via load_inline (nvcc 13.3 + MSVC).

Semantic note (honest): the retained set may differ from exact top-k by a
couple of elements near the threshold — that IS the mechanism, and the
overlap benchmark quantifies it.
"""
import os, time
import torch
from torch.utils.cpp_extension import load_inline

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"   # RTX 4070S = Ada; must be set BEFORE load_inline

cpp_src = "std::vector<torch::Tensor> kwta_forward(torch::Tensor x, int64_t k);"

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kwta_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            bool* __restrict__ mask,
                            int d, int k, long long total_rows)
{
    long long row = (long long)blockIdx.x;
    if (row >= total_rows) return;
    const float* xr = x + row * (long long)d;
    float* yr = y + row * (long long)d;
    bool* mr = mask + row * (long long)d;

    extern __shared__ float sv[];                 // d floats (row cache)
    __shared__ float red[256][2];
    __shared__ float s_theta;
    __shared__ int s_cnt;

    float lmax = -1e30f, lsum = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = xr[j];
        sv[j] = v;
        lmax = fmaxf(lmax, v);
        lsum += v;
    }
    red[threadIdx.x][0] = lmax;
    red[threadIdx.x][1] = lsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x][0] = fmaxf(red[threadIdx.x][0], red[threadIdx.x + s][0]);
            red[threadIdx.x][1] += red[threadIdx.x + s][1];
        }
        __syncthreads();
    }
    float theta = fmaxf(red[0][1] / (float)d, 0.f);
    float mabs = fmaxf(fabsf(red[0][0]), 1.f);
    if (threadIdx.x == 0) s_cnt = 0;

    // iterative threshold adaptation (Krotov-Hopfield style), <= 8 rounds
    for (int it = 0; it < 8; ++it) {
        int lcnt = 0;
        for (int j = threadIdx.x; j < d; j += blockDim.x)
            lcnt += (sv[j] > theta) ? 1 : 0;
        if (threadIdx.x == 0) s_cnt = 0;
        __syncthreads();
        atomicAdd(&s_cnt, lcnt);
        __syncthreads();
        int cnt = s_cnt;
        if (cnt == k) break;
        float step = (float)(cnt - k) / (float)k * 0.04f * mabs;
        theta += step;
        if (theta < 0.f) theta = 0.f;
        __syncthreads();
    }
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        bool keep = (sv[j] > theta);
        yr[j] = keep ? sv[j] : 0.f;
        mr[j] = keep;
    }
}

std::vector<torch::Tensor> kwta_forward(torch::Tensor x, int64_t k)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "fp32 cuda tensor required");
    auto xc = x.contiguous();
    auto y = torch::empty_like(xc);
    auto mask = torch::empty(xc.sizes(), xc.options().dtype(torch::kBool));
    long long rows = xc.numel() / xc.size(-1);
    int d = xc.size(-1);
    int threads = 256;
    size_t smem = (size_t)d * sizeof(float);
    dim3 grid((unsigned)rows);
    kwta_kernel<<<grid, threads, smem>>>(xc.data_ptr<float>(), y.data_ptr<float>(),
                                         mask.data_ptr<bool>(), d, (int)k, rows);
    return {y, mask};
}
"""

mod = None


def _get_mod():
    """Lazy load: try prebuilt .pyd first (no MSVC needed), else compile."""
    global mod
    if mod is None:
        import importlib.util
        pyd = (r"C:\Users\user\AppData\Local\torch_extensions"
               r"\torch_extensions\Cache\py313_cu118\adaptive_kwta_v1"
               r"\adaptive_kwta_v1.pyd")
        if os.path.exists(pyd):
            spec = importlib.util.spec_from_file_location("adaptive_kwta_v1", pyd)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        mod = load_inline(
            name="adaptive_kwta_v1",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["kwta_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor", "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"],
            verbose=False,
        )
    return mod


class AdaptiveKWTA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        y, mask = _get_mod().kwta_forward(x, k)
        ctx.save_for_backward(mask)       # explicit bool mask (a kept value
        return y                          # could itself be 0.0)

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad * mask.to(grad.dtype), None


def adaptive_kwta(x, k_frac=0.10):
    d = x.shape[-1]
    k = max(1, int(d * k_frac))
    return AdaptiveKWTA.apply(x, k)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 256, 768, device="cuda")
    k = int(768 * 0.10)

    # correctness: overlap of retained set vs exact top-k
    y_ad = adaptive_kwta(x, 0.10)
    kth = torch.kthvalue(x, 768 - k + 1, dim=-1, keepdim=True).values
    y_exact = x * (x >= kth)
    n_ad = int((y_ad != 0).sum())
    overlap = int(((y_ad != 0) & (y_exact != 0)).sum())
    print(f"retained: adaptive={n_ad}, exact={int((y_exact != 0).sum())}, "
          f"overlap={overlap} ({overlap / max(n_ad, 1):.1%})")
    print(f"values equal where both kept: "
          f"{torch.allclose(y_ad[(y_ad != 0) & (y_exact != 0)], y_exact[(y_ad != 0) & (y_exact != 0)])}")

    # benchmark: 100 iterations each
    def bench(fn, iters=100):
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000

    t_ad = bench(lambda: adaptive_kwta(x, 0.10))
    t_exact = bench(lambda: x * (x >= torch.kthvalue(x, 768 - k + 1, dim=-1, keepdim=True).values))
    t_topk = bench(lambda: torch.topk(x, k, dim=-1).values)
    print(f"adaptive CUDA: {t_ad:.3f} ms | kthvalue fused: {t_exact:.3f} ms | torch.topk: {t_topk:.3f} ms")
