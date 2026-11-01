# qwen4exp's kernel modules (ADR-0024 item 3), named qwen4exp*.

engine_add_kernel_module(qwen4exp SOURCE kernels/models/qwen4exp/layer.cl
                         OPTIONS -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp -I${CMAKE_SOURCE_DIR}/kernels/models/qwen4exp)
engine_add_kernel_module(qwen4exp_prompt SOURCE kernels/models/qwen4exp/prompt.cl
                         OPTIONS -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp -I${CMAKE_SOURCE_DIR}/kernels/models/qwen4exp)
engine_add_kernel_module(qwen4exp_attn SOURCE kernels/models/qwen4exp/prompt_attn.cl
                         OPTIONS -cl-intel-256-GRF-per-thread -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
engine_add_kernel_module(qwen4exp_gdn SOURCE kernels/models/qwen4exp/prompt_gdn.cl
                         OPTIONS -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
# The DFlash 2 drafter's kernels between its products (core/models/qwen4exp/dflash.cpp).
engine_add_kernel_module(qwen4exp_dflash SOURCE kernels/models/qwen4exp/dflash.cl
                         OPTIONS -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
# The image encoder's steps between its products (docs/spec/MODEL_VISION.md).
engine_add_kernel_module(qwen4exp_vision SOURCE kernels/models/qwen4exp/vision.cl)
engine_add_kernel_module(qwen4exp_vision_attn SOURCE kernels/models/qwen4exp/vision.cl OPTIONS -cl-intel-256-GRF-per-thread)
