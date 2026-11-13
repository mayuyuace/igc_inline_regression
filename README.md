# IGC 26.44 regression: lambda no longer inlined → by-ref arrays go to private memory → 12x slowdown

## Summary

With the **same SPIR-V**, IGC from `intel-opencl-icd 26.44.40863` no longer
inlines a small lambda that takes two register arrays by reference
(`float (&)[8]`, `int (&)[8]`). IGC `26.43.40581` inlines it.

Because the lambda is outlined, the arrays passed to it must be addressable, so
they are demoted to private memory (`private_size: 5888` bytes per work-item,
stateless global memory). The kernel's hot loop then does roughly twice as many
global loads/stores, and runtime regresses by up to **12x**.

No source change and no SYCL front-end change is involved: swapping only the
IGC used to finalize the SPIR-V reproduces the difference (see the 2x2 matrix
below).

## Environment

| | Good | Bad |
|---|---|---|
| Driver (`intel-opencl-icd` / IGC) | **26.43.40581** | **26.44.40863** |
| Docker image | `pytorch-ipex-spr:baseimage_main_2675_1023` | `pytorch-ipex-spr:baseimage_main_2850_1103_V2` |
| icpx (irrelevant, see matrix) | 2026.3.0 (20260929) | 2026.4.0 (20261030) |
| Device | Intel(R) Graphics [0x674f] (CRI) | same |
| Source | vllm-xpu-kernels commit `9daf33f0` | same |
| Build | `-O3 -fsycl -fsycl-targets=spir64` (JIT) | same |

## Reproduce (offline, ~1 min, no GPU needed)

```bash
./repro.sh            # optional: ./repro.sh <device_id>, default 0x674f
```

It compiles `topk_per_row.spv` with the installed `ocloc` and checks kernel
`_ZTSN4vllm28top_k_per_row_prefill_kernelILi512ELb0EEE`.

Output with IGC 26.43.40581 (expected):
```
non-inlined loadChunk calls    : 0
private_size (bytes / WI)      : 0
load.ugm / store.ugm in ISA    : 240 / 59
RESULT: GOOD - lambda inlined, no private memory
```

Output with IGC 26.44.40863 (regression):
```
non-inlined loadChunk calls    : 80
private_size (bytes / WI)      : 5888
load.ugm / store.ugm in ISA    : 463 / 186
RESULT: BAD  - lambda not inlined, by-ref arrays demoted to private memory
```

## Evidence

### 2x2 matrix: SPIR-V producer (icpx) x finalizer (IGC), prefill kernel

| SPIR-V from \ IGC | 26.43.40581 | 26.44.40863 |
|---|---|---|
| icpx 2026.3 | 1480 sends, no private | **1860 sends, private** |
| icpx 2026.4 | 1480 sends, no private | **1860 sends, private** |

The result depends only on IGC.

### Non-inlined function (IGC 26.44 `*_optimized.ll`)

```
internal spir_func void
@_ZZN4vllmL16topKPerRowReduceILi512ELi512ELi2048ELi7ELi8ELi128ELb0ELi2147483647EEE...ENKUliRA8_fRA8_iE_clEiSC_SE_(
    ptr ..., i32 ..., ptr ... dereferenceable(32) %2, ptr ... )
```
That is `topKPerRowReduce<...>::{lambda(int, float(&)[8], int(&)[8])}::operator()`,
with 80 call sites. With IGC 26.43 there are 0 calls and 0 remaining `alloca`s.
With IGC 26.44, 32 `alloca`s survive, including `[8 x float]` and `[8 x i32]`.

### zeinfo (prefill kernel)

| | IGC 26.43 | IGC 26.44 |
|---|---|---|
| `private_size` | — (0) | **5888** |
| `per_thread_memory_buffers: private_space` | no | **yes** |
| `grf_count` | 160 | 160 |
| GRF spill | none | none |

### Final ISA (prefill kernel)

| | IGC 26.43 | IGC 26.44 |
|---|---|---|
| `load.ugm` | 240 | **463** |
| `store.ugm` | 59 | **186** |
| `load.ugm.d64.a64` (private) | 0 | 86 |
| `store.ugm.d32.a64.wb` (private) | 0 | 88 |

Reference dumps for both drivers are in `ref_dumps/igc_2675` (26.43) and
`ref_dumps/igc_2850` (26.44): `*_optimized.ll`, `*.zeinfo`, and
`prefill_kernel.asm`.

### Runtime impact (same vllm-xpu-kernels source, measured with the PyTorch profiler)

| Kernel | IGC 26.43 | IGC 26.44 | IGC 26.44 + `IGC_FunctionControl=1` |
|---|---|---|---|
| `top_k_per_row_prefill_kernel<512,false>` (2560 x 51200) | 762 us | **9203 us** | 762 us |
| `top_k_per_row_decode_kernel<512,false,false,false>` | 23.7 us | **63.4 us** | 23.7 us |

`runtime_perf.py` measures this on a GPU. It needs PyTorch XPU and the
vllm-xpu-kernels wheel built from commit `9daf33f0` without the workaround
below.

## Workarounds (confirmed)

1. `IGC_FunctionControl=1` (force inline everything). This restores performance
   but is global and heavy-handed.
2. In source, mark the lambda `__attribute__((always_inline))`. This restores
   762 us / 23.8 us with no env var:
   ```cpp
   auto loadChunk =
       [&](int chunk, float (&val)[kMaxRegItems], int (&off)[kMaxRegItems])
           __attribute__((always_inline)) { ... };
   ```

## Ask for the IGC team

- Which 26.43 → 26.44 change altered the inlining heuristic for this case?
- Please consider inline cost/benefit heuristics that account for the fact that
  outlining a callee that takes private arrays by reference forces
  private-memory demotion (SROA/mem2reg blocked). In this kernel, that is
  much more expensive than the code-size savings from outlining.

## Files

| File | Purpose |
|---|---|
| `repro.sh` | Offline reproducer (ocloc only) |
| `topk_per_row.spv` | SPIR-V of vllm-xpu-kernels `csrc/topk_per_row.cpp` (no always_inline) |
| `ref_dumps/igc_2675/` | IGC 26.43 good dumps |
| `ref_dumps/igc_2850/` | IGC 26.44 bad dumps |
| `runtime_perf.py` | Optional GPU timing (needs torch + vllm-xpu-kernels) |
