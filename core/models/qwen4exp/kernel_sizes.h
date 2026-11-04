// The sizes qwen4exp's kernel modules are built for. A kernel gets them through layout.h; shape.cpp compares a file with
// them, and tests that drive kernels include this file. Model code in the core reads Shape (ADR-0064).
#ifndef QWEN4EXP_KERNEL_SIZES_H
#define QWEN4EXP_KERNEL_SIZES_H

#define Q_ATT_D 256
#define Q_ATT_H 24  // query heads
#define Q_ATT_KV 2  // KV heads; query head h reads KV head h / 12
#define Q_GDN_CONV 4
#define Q_GDN_D 128  // head size
#define Q_HC 4  // residual streams
#define Q_HC_LR 320
#define Q_USED 10

#endif
