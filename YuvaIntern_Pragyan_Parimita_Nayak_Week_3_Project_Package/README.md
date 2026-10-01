# YuvaIntern Week 3 - Pragyan Parimita Nayak

## Project
Real-Time Retail Shelf Object Detection and Counting

## Optimization
Structured channel pruning (25% removed) followed by three fine-tuning epochs.
A TorchScript representation was also produced for inference deployment.

## Why this optimization
Week 2 identified a compact detector as the baseline. Structured pruning was selected
because removing complete convolution channels reduces the actual tensor dimensions,
unlike unstructured sparsity which may not improve normal CPU inference.

## Benchmark
Baseline median latency: 1.902 ms
Pruned eager median latency: 1.654 ms
Pruned TorchScript median latency: 0.771 ms

Baseline median-based FPS: 525.78
Pruned eager median-based FPS: 604.73
Pruned TorchScript median-based FPS: 1296.97

Parameter reduction: 43.48%

The benchmark uses the same 128x128 synthetic validation distribution as Week 2.
No SKU-110K benchmark metric is claimed here.
