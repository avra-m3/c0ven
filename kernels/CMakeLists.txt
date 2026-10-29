# Kernel modules, built ahead of time by ocloc (cmake/kernels.cmake).

engine_add_kernel_module(smoke SOURCE kernels/smoke.cl)
engine_add_kernel_module(common SOURCE kernels/common.cl)
engine_add_kernel_module(gemm_f16 SOURCE kernels/gemm.cl OPTIONS -cl-intel-256-GRF-per-thread)
engine_add_kernel_module(gemm_bf16 SOURCE kernels/gemm.cl OPTIONS -DBF16 -cl-intel-256-GRF-per-thread)
# bf16 with tiles of 4 tokens, for the prompt's products whose K or N is short (kernels/gemm.cl).
engine_add_kernel_module(gemm_bf16_t4n2 SOURCE kernels/gemm.cl
                         OPTIONS -DBF16 -DTM=4 -DTN=2 -cl-intel-256-GRF-per-thread)
engine_add_kernel_module(gemm_bf16_t4n1 SOURCE kernels/gemm.cl
                         OPTIONS -DBF16 -DTM=4 -DTN=1 -cl-intel-256-GRF-per-thread)
# bf16, 32 tokens by 32 rows a sub-group, 11 sub-groups a work-group: one work-group covers the hyper-connections' 324
# columns (kernels/gemm.cl).
engine_add_kernel_module(gemm_bf16_t4n2s11 SOURCE kernels/gemm.cl
                         OPTIONS -DBF16 -DTM=4 -DTN=2 -DGEMM_SG=11 -cl-intel-256-GRF-per-thread)
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

# One matrix-vector module per weight type and lanes per row: matvec_<type>_l<lanes>.
foreach(type Q8_0 Q4_0 Q5_0 Q2_0 IQ4_NL IQ4_XS Q3_K Q4_K Q5_K Q6_K IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S)
  string(TOLOWER ${type} lower)
  foreach(lanes 16 8 4)
    engine_add_kernel_module(matvec_${lower}_l${lanes} SOURCE kernels/quant/matvec.cl
                             OPTIONS -DQTYPE=QT_${type} -DLANES=${lanes} -I${CMAKE_SOURCE_DIR}/kernels/quant)
  endforeach()
endforeach()

# The prompt path's quantised products, one module per type in the runs layout (qgemm_<type>), and for the experts'
# types also in the file's layout (qgemm_<type>_file), for the experts streamed to the cards.
foreach(type Q8_0 Q4_0 Q5_0 Q2_0 IQ4_NL IQ4_XS Q3_K Q4_K Q5_K Q6_K IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S)
  string(TOLOWER ${type} lower)
  engine_add_kernel_module(qgemm_${lower} SOURCE kernels/quant/qgemm.cl
                           OPTIONS -DQTYPE=QT_${type} -cl-intel-256-GRF-per-thread -I${CMAKE_SOURCE_DIR}/kernels/quant)
endforeach()
foreach(type Q8_0 Q2_0 IQ4_NL IQ4_XS IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S)
  string(TOLOWER ${type} lower)
  engine_add_kernel_module(qgemm_${lower}_file SOURCE kernels/quant/qgemm.cl
                           OPTIONS -DQTYPE=QT_${type} -DLAYOUT_FILE -cl-intel-256-GRF-per-thread -I${CMAKE_SOURCE_DIR}/kernels/quant)
endforeach()

# The decode step's expert products for the experts' types, one module per type and largest group: moe_<type>_g<a>.
foreach(type IQ2_S IQ3_XXS IQ3_S IQ4_XS IQ4_NL Q2_0 Q8_0 IQ2_XXS IQ2_XS IQ1_M)
  string(TOLOWER ${type} lower)
  foreach(a 1 2 3 4 6 8)
    engine_add_kernel_module(moe_${lower}_g${a} SOURCE kernels/quant/moe.cl
                             OPTIONS -DQTYPE=QT_${type} -DGROUP=${a} -I${CMAKE_SOURCE_DIR}/kernels/quant)
  endforeach()
endforeach()
