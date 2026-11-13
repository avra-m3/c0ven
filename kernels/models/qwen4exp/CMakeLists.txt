# qwen4exp's kernel modules (ADR-0024 item 3), named qwen4exp*.

# The modules that route experts: one for each count a file can have, which core/models/qwen4exp/shape.cpp picks from
# (ADR-0063). The draft layer's checkpoint always has 512.
foreach(experts 512 256)
  if(experts EQUAL 512)
    set(suffix "")
  else()
    set(suffix "_e${experts}")
  endif()
  engine_add_kernel_module(qwen4exp${suffix} SOURCE kernels/models/qwen4exp/layer.cl
                           OPTIONS -DQ_EXPERTS=${experts} -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp
                                   -I${CMAKE_SOURCE_DIR}/kernels/models/qwen4exp)
  # The same for the decode steps of up to 8 rows (STEP_NARROW_ROWS, shared/step.h).
  engine_add_kernel_module(qwen4exp${suffix}_r8 SOURCE kernels/models/qwen4exp/layer.cl
                           OPTIONS -DQ_EXPERTS=${experts} -DSTEP_ROWS=8 -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp
                                   -I${CMAKE_SOURCE_DIR}/kernels/models/qwen4exp)
  engine_add_kernel_module(qwen4exp_prompt${suffix} SOURCE kernels/models/qwen4exp/prompt.cl
                           OPTIONS -DQ_EXPERTS=${experts} -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp
                                   -I${CMAKE_SOURCE_DIR}/kernels/models/qwen4exp)
endforeach()
engine_add_kernel_module(qwen4exp_attn SOURCE kernels/models/qwen4exp/prompt_attn.cl
                         OPTIONS -cl-intel-256-GRF-per-thread -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
engine_add_kernel_module(qwen4exp_gdn SOURCE kernels/models/qwen4exp/prompt_gdn.cl
                         OPTIONS -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
# The DFlash 2 drafter's kernels between its products (core/models/qwen4exp/dflash.cpp).
engine_add_kernel_module(qwen4exp_dflash SOURCE kernels/models/qwen4exp/dflash.cl
                         OPTIONS -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
# The image encoder's steps between its products (docs/spec/MODEL_VISION.md).
engine_add_kernel_module(qwen4exp_vision SOURCE kernels/models/qwen4exp/vision.cl
                         OPTIONS -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
engine_add_kernel_module(qwen4exp_vision_attn SOURCE kernels/models/qwen4exp/vision.cl
                         OPTIONS -cl-intel-256-GRF-per-thread -I${CMAKE_SOURCE_DIR}/core/models/qwen4exp)
