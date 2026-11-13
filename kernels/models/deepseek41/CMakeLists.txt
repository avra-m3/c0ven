# deepseek41's kernel modules (ADR-0024 item 3, ADR-0080), named deepseek41*. Each reads the layout the core shares
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
# The drafter's layers: the router and the combine at its expert counts, and attn_output_a in Q8_0.
engine_add_kernel_module(deepseek41_dsp SOURCE kernels/models/deepseek41/layer.cl
                         OPTIONS -DD41_DRAFTER ${DEEPSEEK41_KERNEL_INCLUDES})
engine_add_kernel_module(deepseek41_og_q8_0 SOURCE kernels/models/deepseek41/output.cl
                         OPTIONS -DQTYPE=QT_Q8_0 -DLANES=4 -I${CMAKE_SOURCE_DIR}/kernels/quant
                         ${DEEPSEEK41_KERNEL_INCLUDES})
# The prompt chunk path (core/models/deepseek41/prompt.cpp): attention with a window cache as long as a chunk and a
# window (D41_RING_WIDE in layout.h, which this number must equal), and the kernels around its products.
engine_add_kernel_module(deepseek41_attn_wide SOURCE kernels/models/deepseek41/attn.cl
                         OPTIONS -DD41_RING=4480 ${DEEPSEEK41_KERNEL_INCLUDES})
engine_add_kernel_module(deepseek41_prompt SOURCE kernels/models/deepseek41/prompt.cl
                         OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
set(deepseek41_modules deepseek41 deepseek41_attn deepseek41_attn_wide deepseek41_prompt deepseek41_check deepseek41_og
    deepseek41_dsp deepseek41_og_q8_0)

# The routed experts' products at the file's types, one module per kernel, type and largest group:
# deepseek41_moe_gate_up_<type>_g<a> and deepseek41_moe_down_<type>_g<a>. The down modules read the codebook through
# __constant except IQ2_XXS's at group size 1 (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-moe-step-sweep).
# MXFP4 is the drafter's experts' type.
set(deepseek41_moe_options -DMOE_CLASSES -I${CMAKE_SOURCE_DIR}/kernels/quant ${DEEPSEEK41_KERNEL_INCLUDES})
foreach(a 1 2 3 4 6 8)
  foreach(type IQ1_S IQ1_M IQ2_XXS MXFP4)
    string(TOLOWER ${type} lower)
    engine_add_kernel_module(deepseek41_moe_gate_up_${lower}_g${a} SOURCE kernels/quant/moe.cl
                             OPTIONS -DQTYPE=QT_${type} -DGROUP=${a} -DCL_GATE_UP_ONLY ${deepseek41_moe_options})
    list(APPEND deepseek41_modules deepseek41_moe_gate_up_${lower}_g${a})
  endforeach()
  foreach(type IQ1_S IQ1_M IQ2_XXS IQ2_S MXFP4)
    string(TOLOWER ${type} lower)
    set(grid -DGRID_CONSTANT)
    if(type STREQUAL IQ2_XXS AND a EQUAL 1)
      set(grid "")
    endif()
    engine_add_kernel_module(deepseek41_moe_down_${lower}_g${a} SOURCE kernels/quant/moe.cl
                             OPTIONS -DQTYPE=QT_${type} -DGROUP=${a} -DCL_DOWN_ONLY ${grid}
                             ${deepseek41_moe_options})
    list(APPEND deepseek41_modules deepseek41_moe_down_${lower}_g${a})
  endforeach()
endforeach()
# ds41_moe_test --sweep: each kernel by type, largest group, lanes, rows a sub-group and the codebook's place, in
# deepseek41_moe_sweep_<kernel>_<type>_g<a>_l<lanes>_r<rows>_<place>. 4 lanes at group size 1 only.
if(ENGINE_KERNEL_SWEEPS)
  foreach(kernel gate_up down)
    if(kernel STREQUAL gate_up)
      set(types IQ1_S IQ1_M IQ2_XXS)
      set(only -DCL_GATE_UP_ONLY)
    else()
      set(types IQ1_S IQ1_M IQ2_XXS IQ2_S)
      set(only -DCL_DOWN_ONLY)
    endif()
    foreach(type ${types})
      string(TOLOWER ${type} lower)
      foreach(a 1 2 3 4 6 8)
        set(shapes 16:1 16:2 16:4 8:2 8:4)
        if(a EQUAL 1)
          list(APPEND shapes 4:4)
        endif()
        foreach(shape ${shapes})
          string(REPLACE ":" ";" parts ${shape})
          list(GET parts 0 lanes)
          list(GET parts 1 rows)
          foreach(place local constant)
            set(grid "")
            if(place STREQUAL constant)
              set(grid -DGRID_CONSTANT)
            endif()
            set(module deepseek41_moe_sweep_${kernel}_${lower}_g${a}_l${lanes}_r${rows}_${place})
            engine_add_kernel_module(${module} SOURCE kernels/quant/moe.cl
                                     OPTIONS -DQTYPE=QT_${type} -DGROUP=${a} ${only} -DGU_LANES=${lanes}
                                     -DDOWN_LANES=${lanes} -DCL_ROWS=${rows} ${grid} ${deepseek41_moe_options})
            list(APPEND deepseek41_modules ${module})
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endif()

# engine_add_kernel_module depends on the headers under kernels/ and shared/; these two lie outside them.
foreach(module ${deepseek41_modules})
  foreach(ip ${ENGINE_KERNEL_IPS})
    add_custom_command(OUTPUT ${ENGINE_KERNEL_DIR}/${ip}/${module}.bin APPEND
                       DEPENDS ${CMAKE_SOURCE_DIR}/core/models/deepseek41/layout.h
                               ${CMAKE_SOURCE_DIR}/core/models/deepseek41/rope.h)
  endforeach()
endforeach()
