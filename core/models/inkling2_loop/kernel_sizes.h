// Compile-time bounds shared with the OpenCL kernels; other model dimensions live in Shape.
#ifndef ENGINE_INKLING2_KERNEL_SIZES_H
#define ENGINE_INKLING2_KERNEL_SIZES_H

#define I2_EMBD 2304
#define I2_HEAD_D 128
#define I2_REL_D 16
#define I2_REL_EXT 1024
#define I2_CONV 4
#define I2_EXPERTS 128
#define I2_USED 8
#define I2_GATE_RANK 64
#define I2_RES_MAX 8
#define I2_LOOP_MSGS 4
#define I2_KV_DATA_BYTES (I2_HEAD_D * 2)

#endif
