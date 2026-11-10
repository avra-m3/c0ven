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
# Each architecture's own modules (ADR-0061).
add_subdirectory(models/qwen4exp)

# Every quantised weight type the engine reads.
set(ENGINE_QUANT_TYPES Q8_0 Q4_0 Q5_0 Q5_1 Q2_0 IQ4_NL IQ4_XS Q2_K Q3_K Q4_K Q5_K Q6_K IQ1_S IQ1_M IQ2_XXS IQ2_XS
    IQ2_S IQ3_XXS IQ3_S)

# One matrix-vector module per weight type and lanes per row: matvec_<type>_l<lanes>, up to 8 tokens a weight read,
# and matvec_<type>_l<lanes>_t16, up to 16, for the decode steps of more than 8 rows.
foreach(type ${ENGINE_QUANT_TYPES})
  string(TOLOWER ${type} lower)
  foreach(lanes 16 8 4)
    engine_add_kernel_module(matvec_${lower}_l${lanes} SOURCE kernels/quant/matvec.cl
                             OPTIONS -DQTYPE=QT_${type} -DLANES=${lanes} -I${CMAKE_SOURCE_DIR}/kernels/quant)
    engine_add_kernel_module(matvec_${lower}_l${lanes}_t16 SOURCE kernels/quant/matvec.cl
                             OPTIONS -DQTYPE=QT_${type} -DLANES=${lanes} -DMAXT=16 -I${CMAKE_SOURCE_DIR}/kernels/quant)
  endforeach()
endforeach()

# The prompt path's quantised products, one module per type in the runs layout (qgemm_<type>), and also in the file's
# layout (qgemm_<type>_file), for the experts streamed to the cards: any type may be an expert type.
foreach(type ${ENGINE_QUANT_TYPES})
  string(TOLOWER ${type} lower)
  engine_add_kernel_module(qgemm_${lower} SOURCE kernels/quant/qgemm.cl
                           OPTIONS -DQTYPE=QT_${type} -cl-intel-256-GRF-per-thread -I${CMAKE_SOURCE_DIR}/kernels/quant)
  engine_add_kernel_module(qgemm_${lower}_file SOURCE kernels/quant/qgemm.cl
                           OPTIONS -DQTYPE=QT_${type} -DLAYOUT_FILE -cl-intel-256-GRF-per-thread -I${CMAKE_SOURCE_DIR}/kernels/quant)
endforeach()

# The decode step's expert products for the experts' types, one module per type and largest group: moe_<type>_g<a>.
foreach(type IQ2_S IQ3_XXS IQ3_S IQ4_XS IQ4_NL Q2_0 Q8_0 Q5_1 IQ2_XXS IQ2_XS IQ1_S IQ1_M Q2_K)
  string(TOLOWER ${type} lower)
  foreach(a 1 2 3 4 6 8)
    engine_add_kernel_module(moe_${lower}_g${a} SOURCE kernels/quant/moe.cl
                             OPTIONS -DQTYPE=QT_${type} -DGROUP=${a} -I${CMAKE_SOURCE_DIR}/kernels/quant)
  endforeach()
endforeach()

# The SwiGLU limit (kernels/include/swiglu.h) at 10, for core/tests/swiglu_limit_test.cpp: one variant of each module
# that computes it, for IQ4_XS gate and up.
engine_add_kernel_module(common_limit10 SOURCE kernels/common.cl OPTIONS -DSWIGLU_LIMIT=10.0f)
engine_add_kernel_module(matvec_iq4_xs_l16_limit10 SOURCE kernels/quant/matvec.cl
                         OPTIONS -DQTYPE=QT_IQ4_XS -DLANES=16 -DSWIGLU_LIMIT=10.0f -I${CMAKE_SOURCE_DIR}/kernels/quant)
engine_add_kernel_module(moe_iq4_xs_g2_limit10 SOURCE kernels/quant/moe.cl
                         OPTIONS -DQTYPE=QT_IQ4_XS -DGROUP=2 -DSWIGLU_LIMIT=10.0f -I${CMAKE_SOURCE_DIR}/kernels/quant)
engine_add_kernel_module(qgemm_iq4_xs_file_limit10 SOURCE kernels/quant/qgemm.cl
                         OPTIONS -DQTYPE=QT_IQ4_XS -DLAYOUT_FILE -DSWIGLU_LIMIT=10.0f -cl-intel-256-GRF-per-thread
                         -I${CMAKE_SOURCE_DIR}/kernels/quant)
