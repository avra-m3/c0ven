// The sizes qwen4exp's kernel modules are built for. A kernel gets them through layout.h; shape.cpp compares a file with
// them, and tests that drive kernels include this file. Model code in the core reads Shape (ADR-0064).
#ifndef QWEN4EXP_KERNEL_SIZES_H
#define QWEN4EXP_KERNEL_SIZES_H

#define Q_ATT_D 256
#define Q_ATT_H 24  // query heads
#define Q_ATT_KV 2  // KV heads; query head h reads KV head h / 12
#define Q_EMBD 2560
#define Q_EPS 1e-6f
#define Q_GDN_CONV 4
#define Q_GDN_D 128  // head size
#define Q_HC 4  // residual streams
#define Q_HC_LR 320
#define Q_IDX_BLOCK 4  // positions per pooled key
#define Q_IDX_D 128
#define Q_IDX_H 4  // indexer query heads
#define Q_IDX_TOPK 2048
#define Q_PLE_DIL 3  // the PLE conv's dilation (the n-gram size)
#define Q_ROPE_BASE 10000000.0f
#define Q_ROT 64  // rotated dimensions per head, NeoX pairs (i, i + 32)
#define Q_USED 10

#endif
