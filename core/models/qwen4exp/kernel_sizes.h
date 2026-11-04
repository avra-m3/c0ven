// The sizes qwen4exp's kernel modules are built for. A kernel gets them through layout.h; shape.cpp compares a file with
// them, and tests that drive kernels include this file. Model code in the core reads Shape (ADR-0064).
#ifndef QWEN4EXP_KERNEL_SIZES_H
#define QWEN4EXP_KERNEL_SIZES_H

#define Q_USED 10

#endif
