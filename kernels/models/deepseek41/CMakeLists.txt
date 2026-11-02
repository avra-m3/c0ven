# deepseek41's kernel modules (ADR-0024 item 3, ADR-0062), named deepseek41*. Each reads the layout the core shares
# with it (core/models/deepseek41/layout.h, rope.h) and this folder's headers.

set(DEEPSEEK41_KERNEL_INCLUDES -I${CMAKE_SOURCE_DIR}/core/models/deepseek41
    -I${CMAKE_SOURCE_DIR}/kernels/models/deepseek41)

engine_add_kernel_module(deepseek41 SOURCE kernels/models/deepseek41/layer.cl OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
engine_add_kernel_module(deepseek41_attn SOURCE kernels/models/deepseek41/attn.cl OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
# deepseek41's tests only.
engine_add_kernel_module(deepseek41_check SOURCE kernels/models/deepseek41/check.cl
                         OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
# The attention output product, attn_output_a in Q5_K, at layout.h's D41_OG_LANES.
engine_add_kernel_module(deepseek41_og SOURCE kernels/models/deepseek41/output.cl
                         OPTIONS -DQTYPE=QT_Q5_K -DLANES=4 -I${CMAKE_SOURCE_DIR}/kernels/quant
                         ${DEEPSEEK41_KERNEL_INCLUDES})
set(deepseek41_modules deepseek41 deepseek41_attn deepseek41_check deepseek41_og)

# The routed experts' products at the file's types, one module per kernel, type and largest group:
# deepseek41_moe_gate_up_<type>_g<a> and deepseek41_moe_down_<type>_g<a>. The down modules read the codebook through
# __constant except IQ2_XXS's, as docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-routed-experts says.
set(deepseek41_moe_options -DMOE_CLASSES -I${CMAKE_SOURCE_DIR}/kernels/quant ${DEEPSEEK41_KERNEL_INCLUDES})
foreach(a 1 2 3 4 6 8)
  foreach(type IQ1_S IQ1_M IQ2_XXS)
    string(TOLOWER ${type} lower)
    engine_add_kernel_module(deepseek41_moe_gate_up_${lower}_g${a} SOURCE kernels/quant/moe.cl
                             OPTIONS -DQTYPE=QT_${type} -DGROUP=${a} -DCL_GATE_UP_ONLY ${deepseek41_moe_options})
    list(APPEND deepseek41_modules deepseek41_moe_gate_up_${lower}_g${a})
  endforeach()
  foreach(type IQ1_S IQ1_M IQ2_XXS IQ2_S)
    string(TOLOWER ${type} lower)
    set(grid -DGRID_CONSTANT)
    if(type STREQUAL IQ2_XXS)
      set(grid "")
    endif()
    engine_add_kernel_module(deepseek41_moe_down_${lower}_g${a} SOURCE kernels/quant/moe.cl
                             OPTIONS -DQTYPE=QT_${type} -DGROUP=${a} -DCL_DOWN_ONLY ${grid}
                             ${deepseek41_moe_options})
    list(APPEND deepseek41_modules deepseek41_moe_down_${lower}_g${a})
  endforeach()
endforeach()
# ds41_moe_test's sweep at group size 1, as lanes:rows a sub-group:name. The name ends in constant when the codebook is
# read through __constant; "local" and "constant" alone are 4 rows a sub-group.
foreach(type IQ1_S IQ1_M IQ2_XXS IQ2_S)
  string(TOLOWER ${type} lower)
  foreach(shape 16:4:local 16:4:constant 16:2:r2 16:2:r2_constant 16:1:r1 8:4:local 8:4:constant 8:2:r2 8:2:r2_constant
                4:4:local 4:4:constant)
    string(REPLACE ":" ";" parts ${shape})
    list(GET parts 0 lanes)
    list(GET parts 1 rows)
    list(GET parts 2 place)
    set(grid "")
    if(place MATCHES constant)
      set(grid -DGRID_CONSTANT)
    endif()
    engine_add_kernel_module(deepseek41_moe_${lower}_l${lanes}_${place} SOURCE kernels/quant/moe.cl
                             OPTIONS -DQTYPE=QT_${type} -DGROUP=1 -DGU_LANES=${lanes} -DDOWN_LANES=${lanes}
                             -DCL_ROWS=${rows} ${grid} ${deepseek41_moe_options})
    list(APPEND deepseek41_modules deepseek41_moe_${lower}_l${lanes}_${place})
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
