# Report 

```
(env) root@601bcab80d10:/home/SigLIP-2# python benchmark.py --batch 1024 --max-patches 256 --dtype bf16 --device cuda
Original -> Avg forward: 2099.66 ms | Avg backward: 2899.58 ms | Peak Mem: 19288.3 MiB
Optimized -> Avg forward: 1273.94 ms | Avg backward: 2971.07 ms | Peak Mem: 12136.7 MiB
```

## Design Choices

# Report 

```
(env) root@601bcab80d10:/home/SigLIP-2# python benchmark.py --batch 1024 --max-patches 256 --dtype bf16 --device cuda
Original -> Avg forward: 2099.66 ms | Avg backward: 2899.58 ms | Peak Mem: 19288.3 MiB
Optimized -> Avg forward: 1273.94 ms | Avg backward: 2971.07 ms | Peak Mem: 12136.7 MiB
```

## Design Choices

I decided to continue with triton, since it's easiest way to integrate kernel with `torch.compile` and makes prototyping easier.

I first tried to compile the model to look at graph breaks. This led me to discovering issues with excessive mask recomputations. 

After fixing the issues, the next step was profiling the model. The activation function in MLP had a huge memory imprint, so to avoid keeping unnecessary copies I decided to recompute parts of it in the backpass.

While working on the fused MLP I got a discovery that affected all of numerical stability in the tests. Since WGMMA accumulator is fp32, if you do ``F.gelu(F.linear(bf16, bf16, bf16))`` you will get downcasting from `fp32` to `bf16` and back to `fp32` as `tanh` in `libdevice` is defined only on `fp32` (actually in toolkit it's `double @__nv_atanh(double %x)` but I am not sure what torch uses for `tanh`). So in our fused MLP we actually gain a bit of precision as can be seen in `numerical_precision.ipynb`.

I also reused FlashAttention LayerNorm implementation due to triton_ops being broken (not supporting tuple returns) and spent too much time debugging here.

I tried to implement seeded dropout, instead of masked one, but it didn't work well with `torch.compile` altough led to about ~1000 MiB decrease.

I fused qkv-layer in attention, so we could exploit more persistent kernel and remove launch overhead.

I tried to keep running the tests to keep numerical stability in check, and tried to profile every module whenever I had the opportunity.

Stream-k was a bit unuseful, as we had bias/activations after allmost every matmul. It was pretty much only performance penalty in my implementations.

I ran some experiments with cuda.graphs and it was pretty impactful, but we are trying to stick to purely kernel performance gains. 

## Thought Process

- Benchmark modules, heavily.
- Look at profiles.
- Look at memory usage.
- Select potential gains (Attention, MLP)
- Look at recomendations for design 
- Look at graph breaks

## More Time

There are still a lot of trade-offs we should explore regarding numerical stability. Triton doesn’t allow controlling ffast-math/FTZ explicitly, and, given time, I’d be happy to explore how to incorporate it via inductor tooling.

Triton API for TMA is still in an experimental stage. So it wasn't very useful digging there.  

Obviously the most important next step would've been incorporating FlashAttention-3 into the model, as flex-attention doesn't support it yet. FA-2 with TMA didn't work so well, due to unusual `head_dim = 72`.

There is a lot of potential performance gains via migrating kernels to cutlass/cute but it's a bit outside of the scope of this project, as `torch.compile` is a bit annoying to deal with when working with external sources.

