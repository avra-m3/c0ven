# deepseek41's kernel modules (ADR-0024 item 3, ADR-0062), named deepseek41*. Each reads the layout the core shares
# with it (core/models/deepseek41/layout.h, rope.h) and this folder's headers.

set(DEEPSEEK41_KERNEL_INCLUDES -I${CMAKE_SOURCE_DIR}/core/models/deepseek41
    -I${CMAKE_SOURCE_DIR}/kernels/models/deepseek41)

engine_add_kernel_module(deepseek41 SOURCE kernels/models/deepseek41/layer.cl OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
engine_add_kernel_module(deepseek41_attn SOURCE kernels/models/deepseek41/attn.cl OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
# deepseek41's tests only.
engine_add_kernel_module(deepseek41_check SOURCE kernels/models/deepseek41/check.cl
                         OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
# The attention output product: a module per attn_output_a's weight types (blocks 0-26 and 27-39) and lanes a row.
set(deepseek41_modules deepseek41 deepseek41_attn deepseek41_check)
foreach(type Q8_0 Q6_K)
  string(TOLOWER ${type} lower)
  foreach(lanes 16 8 4)
    engine_add_kernel_module(deepseek41_og_${lower}_l${lanes} SOURCE kernels/models/deepseek41/output.cl
                             OPTIONS -DQTYPE=QT_${type} -DLANES=${lanes} -I${CMAKE_SOURCE_DIR}/kernels/quant
                             ${DEEPSEEK41_KERNEL_INCLUDES})
    list(APPEND deepseek41_modules deepseek41_og_${lower}_l${lanes})
  endforeach()
endforeach()

# engine_add_kernel_module depends on the headers under kernels/ and shared/; these two lie outside them.
foreach(module ${deepseek41_modules})
  foreach(ip ${ENGINE_KERNEL_IPS})
    add_custom_command(OUTPUT ${ENGINE_KERNEL_DIR}/${ip}/${module}.bin APPEND
                       DEPENDS ${CMAKE_SOURCE_DIR}/core/models/deepseek41/layout.h
                               ${CMAKE_SOURCE_DIR}/core/models/deepseek41/rope.h)
  endforeach()
endforeach()
