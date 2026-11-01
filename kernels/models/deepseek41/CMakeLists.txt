# deepseek41's kernel modules (ADR-0024 item 3, ADR-0062), named deepseek41*. Each reads the layout the core shares
# with it (core/models/deepseek41/layout.h, rope.h) and this folder's headers.

set(DEEPSEEK41_KERNEL_INCLUDES -I${CMAKE_SOURCE_DIR}/core/models/deepseek41
    -I${CMAKE_SOURCE_DIR}/kernels/models/deepseek41)

engine_add_kernel_module(deepseek41 SOURCE kernels/models/deepseek41/layer.cl OPTIONS ${DEEPSEEK41_KERNEL_INCLUDES})
