
import torch
import triton
import triton.language as tl

from triton.backends.intel.compiler import XPUBackend

original_annotate_module = XPUBackend.annotate_module


def force_simd16(cls, module_opts, properties, opt):
    original_annotate_module(module_opts, properties, opt)
    module_opts.min_sg_size = 16


XPUBackend.annotate_module = classmethod(force_simd16)


from triton._C.libtriton import intel
from triton.backends.intel.compiler import XPUBackend

original_make_llir = XPUBackend.make_llir


def force_genisa(cls, mod, metadata, options):
    intel.set_is_lts(mod)
    return original_make_llir(mod, metadata, options)


XPUBackend.make_llir = classmethod(force_genisa)



@triton.jit
def matmul_probe(a_ptr, b_ptr, c_ptr):
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 128)

    a = tl.load(a_ptr + m[:, None] * 128 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * 64 + n[None, :])

    acc = tl.full((64, 64), 0, tl.float32)
    result = tl.dot(a, b, acc)

    tl.store(c_ptr + m[:, None] * 64 + n[None, :], result)


a = torch.randn((64, 128), device="xpu", dtype=torch.bfloat16)
b = torch.randn((128, 64), device="xpu", dtype=torch.bfloat16)
c = torch.empty((64, 64), device="xpu", dtype=torch.float32)

kernel = matmul_probe[(1,)](
    a, b, c,
    num_warps=1,
    warp_size=16,
)

print(kernel.asm["ttgir"])
print(kernel.asm["llir"])
