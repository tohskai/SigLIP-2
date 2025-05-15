# Report 

```
(env) root@601bcab80d10:/home/SigLIP-2# python benchmark.py --batch 1024 --max-patches 256 --dtype bf16 --device cuda
Original -> Avg forward: 2099.66 ms | Avg backward: 2899.58 ms | Peak Mem: 19288.3 MiB
Optimized -> Avg forward: 1273.94 ms | Avg backward: 2971.07 ms | Peak Mem: 12136.7 MiB
```

## Design Choices

I decided to continue with triton, since it's the easiestit's easiest way to integrate kernelsintegrate kernel with `torch.compile` and it also makes prototyping easier.

I first tried to compile the model to look at graph breaks:
```
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks] Graph break in user code at /home/SigLIP-2/model.py:102
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks] Graph Break Reason: Data dependent operator
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   Explanation: Operator `aten._local_scalar_dense.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   Hint: Enable tracing of data-dependent output operators with `torch._dynamo.config.capture_scalar_outputs = True`
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   Developer debug context: aten._local_scalar_dense.default
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks] User code traceback:
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   File "/home/SigLIP-2/benchmark.py", line 231, in <module>
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]     main()
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   File "/home/SigLIP-2/benchmark.py", line 208, in main
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]     benchmark_model(
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   File "/home/SigLIP-2/benchmark.py", line 100, in benchmark_model
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]     out = model(inputs)
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   File "/home/SigLIP-2/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]     return self._call_impl(*args, **kwargs)
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   File "/home/SigLIP-2/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]     return forward_call(*args, **kwargs)
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   File "/home/SigLIP-2/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]     return self._call_impl(*args, **kwargs)
V0515 21:32:02.058000 49093 env/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:556] [0/0] [__graph_breaks]   File "/home/SigLIP-2/env/lib/python3.10/site-packages/torch/nn/modules/module.py
```

This led me to discovering issues with excessive mask recomputations inside the kernel, which I put outside of attention layer, which compiled now without graph breaks. 

After fixing the issues, the next step was profiling the model. The activation function in MLP had a huge memory footprint, so to avoid keeping unnecessary copies I decided to recompute parts of it in the backpass kernel.

While working on the fused MLP I  discovered something that affects numerical stability. Since WGMMA accumulator is `fp32`, if you do ``F.gelu(F.linear(bf16, bf16, bf16))`` you will get downcasting from `fp32` to `bf16` and back to `fp32` as `tanh` in `libdevice` is defined only on `fp32` (actually in toolkit it's `double @__nv_atanh(double %x)` but I am not sure what torch uses for `tanh`). So in our fused MLP we actually gain a bit of precision as can be seen in `numerical_precision.ipynb`.

As I needed to fuse Addition with LayerNorm and potentially do it in-place to use less memory I reused FlashAttention LayerNorm implementation and due to triton_ops wrapper for `torch.compile` being broken (not supporting tuple returns) and spent too much time debugging here.

I tried to implement seeded dropout (we keep only the seed for rng, instead of mask), but it didn't work well with `torch.compile`, but led to about ~1000 MiB decrease. I think it's possible to integrate it, but it will take some time.

I fused qkv-layer in attention, so we could avoid kernel launch overhead.

Stream-k (decomposing gemm over K-axis didn’t lead to any gains as we have bias/activations after allmost every matmul. It was pretty much only performance penalty in my implementations. And there are also known [issues](https://github.com/triton-lang/triton/issues/1393)with it in triton.

I ran some experiments with cuda.graphs and it was pretty impactful, but we are trying to stick to purely kernel performance gains. 

## Thought Process

- Benchmark modules, heavily.
- Look at profiles.
- Look at memory usage.
- Select potential gains (Attention, MLP)
- Look at recomendations for design 
- Look at graph breaks

## More Time

The most important next step would be to incorporate FlashAttention-3 into the model, as flex-attention doesn't support it yet. FA-2 with TMA didn't work so well, due to unusual `head_dim = 72`. (it's good whem dim is divisible by 32, as we can divide tiles in warps)

There are still a lot of trade-offs we should explore regarding numerical stability. Triton doesn’t allow controlling ffast-math/FTZ explicitly, and, given time, I’d be happy to explore how to incorporate it via inductor tooling.

Triton API for TMA is still in an experimental stage. So it wasn't very useful digging there.  

There are still a lot of potential performance gains via migrating kernels to cutlass/cute (more low-level control, tensor cores targeting, better vectorization, warp specialization, hardware-specific optimizations, tile scheduler, easier pipelining) but it's a bit outside of the scope of this project, as `torch.compile` is a bit annoying to deal with when working with external sources.

