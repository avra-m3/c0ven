// The sizes qwen4exp's image encoder kernels are built for (C++20 and OpenCL C). vision.cl takes them; vision.cpp
// compares an encoder file with them, each with the key it comes from (docs/spec/MODEL_VISION.md).
#ifndef QWEN4EXP_VISION_SIZES_H
#define QWEN4EXP_VISION_SIZES_H

#define V_D 1152          // the residual's width: clip.vision.embedding_length
#define V_H 16            // heads: clip.vision.attention.head_count
#define V_FF 4304         // the MLP's width: clip.vision.feed_forward_length
#define V_LAYERS 27       // clip.vision.block_count
#define V_OUT 2560        // the merger's output, the text model's embedding: clip.vision.projection_dim
#define V_SIDE 16         // a patch's side in pixels: clip.vision.patch_size
#define V_MERGE 2         // patches merged into a token along each side: clip.vision.spatial_merge_size
#define V_GRID 48         // the position table's side: v.position_embd.weight has V_GRID x V_GRID rows
#define V_EPS 1e-6f       // clip.vision.attention.layer_norm_epsilon
#define V_PIX_MEAN 0.5f   // each channel's clip.vision.image_mean
#define V_PIX_STD 0.5f    // each channel's clip.vision.image_std

#define V_HD (V_D / V_H)                       // a head's dimensions: 72
#define V_ROT (V_HD / 2)                       // rotated pairs per head: (k, k + 36)
#define V_SEC (V_ROT / 2)                      // pairs turned by the row; the next ones by the column
#define V_PATCH (3 * V_SIDE * V_SIDE)          // 768 bytes: one frame of a patch, 8-bit RGB
#define V_FFP ((V_FF + 31) / 32 * 32)          // the MLP's width in the stages, padded with zeros: 4,320
#ifndef V_HDP
#define V_HDP ((V_HD + 15) / 16 * 16)          // a head in the fp16 Q, K and V, padded with zeros: 80
#endif

#endif
