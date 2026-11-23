# inkling2_loop kernel modules (ADR-0024 item 3).

set(I2_OPTS -I${CMAKE_SOURCE_DIR}/core/models/inkling2_loop
            -I${CMAKE_SOURCE_DIR}/kernels/models/inkling2_loop
            -I${CMAKE_SOURCE_DIR}/kernels/quant)

engine_add_kernel_module(inkling2_layer SOURCE kernels/models/inkling2_loop/layer.cl
                         OPTIONS ${I2_OPTS})
engine_add_kernel_module(inkling2_prompt SOURCE kernels/models/inkling2_loop/prompt.cl
                         OPTIONS ${I2_OPTS})
engine_add_kernel_module(inkling2_prompt_attn SOURCE kernels/models/inkling2_loop/prompt_attn.cl
                         OPTIONS -cl-intel-256-GRF-per-thread ${I2_OPTS})

set(I2_MOE_TYPES Q8_0 IQ4_NL Q4_0 Q5_0 Q5_1 Q4_K Q5_K Q6_K IQ4_XS)
foreach(t ${I2_MOE_TYPES})
  string(TOLOWER ${t} tl)
  foreach(g 1 2 4 8)
    engine_add_kernel_module(inkling2_moe_${tl}_g${g} SOURCE kernels/models/inkling2_loop/moe.cl
                             OPTIONS -DQTYPE=QT_${t} -DGROUP=${g} ${I2_OPTS})
  endforeach()
  engine_add_kernel_module(inkling2_qgemm_${tl} SOURCE kernels/models/inkling2_loop/qgemm_moe.cl
                           OPTIONS -DQTYPE=QT_${t} -cl-intel-256-GRF-per-thread ${I2_OPTS})
endforeach()
