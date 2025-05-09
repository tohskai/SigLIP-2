# Report 

```
(env) root@c3a59061e311:/home/SigLIP-2# python benchmark.py --batch 1024 --max-patch 256 --dtype bf16 --device cuda
Original -> Avg forward: 2120.85 ms | Avg backward: 2823.85 ms | Peak Mem: 19288.3 MiB
Optimized -> Avg forward: 1444.41 ms | Avg backward: 2530.89 ms | Peak Mem: 19237.0 MiB
```

## Design Choices

Since we are heavily constrained by time, I chose triton for easier prototyping.

I first tried to compile the model to look at graph breaks. This led me to discovering issues with excessive mask recomputations. 

After fixing the issues, the next step was profiling the model. I found excessive copies of GELU-computed tensors due to unfused GELU and worked on fixing the problem by writing a simple fused pass. Unfortunately I didn't have enough time for bakcward pass kernel for fused Linear-GELU.

I also reused LayerNorm implementation in LigerKernels.

## Thought Process

- Benchmark models, heavily.
- Look at profiles.
- Look at memory usage.
- Select potential gains (Attention, MLP)
- Look at recomendations for design (LigerKernel primarily)
- Look at graph breaks

## More Time

Some problems arose with numerical stability, due to lack of ability to control flush-to-zero behaviour and some undefined casts (you can write inline_asm but it will break torch.compile). I would have heavily prefered using CUTLASS here.

Since I had a quite a unique opportunity to work with hoppers, I would've also liked to exploit TMA more carefully.

I'd spend more time on incorporating stream-k into MLP at least (i tried, but got big preformance penalty) and swizzling, and exeperiment more with eviction policy.

Incorporate cudagraphs more efficiently, we have a lot of static shapes.

Possibly explore tradeoffs between recomputing things in backward pass and memory usage.
