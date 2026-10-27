# Kernels are OpenCL C, compiled ahead of time by ocloc into native binaries (ADR-0003).
# The source is built with -cl-std=CL3.0: without it ocloc compiles OpenCL C 1.2. Divides and sqrt are rounded
# correctly, as ggml's CPU code rounds them, so a quantizer's amax / 127 gives ggml's bytes (ADR-0051).

find_program(ENGINE_OCLOC ocloc)
if(NOT ENGINE_OCLOC)
  message(FATAL_ERROR "ocloc not found: install intel-ocloc, pinned in docs/adr/0002-pin-the-toolchain.md")
endif()
set(ENGINE_AOT_DEVICE "bmg-g21" CACHE STRING "ocloc -device value (ADR-0002's AOT target)")
set(ENGINE_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "")
file(MAKE_DIRECTORY ${ENGINE_KERNEL_DIR})

# engine_add_kernel_module(<name> SOURCE <file.cl> [OPTIONS <build options>...])
# Writes ${ENGINE_KERNEL_DIR}/<name>.bin. A variant is another module of the same source with other -D options.
# Every header under kernels/ and shared/ is a dependency, and both directories are on the include path.
function(engine_add_kernel_module name)
  cmake_parse_arguments(K "" "SOURCE" "OPTIONS" ${ARGN})
  set(src ${CMAKE_SOURCE_DIR}/${K_SOURCE})
  file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/kernels/*.h ${CMAKE_SOURCE_DIR}/shared/*.h)
  list(JOIN K_OPTIONS " " opts)
  add_custom_command(
    OUTPUT ${ENGINE_KERNEL_DIR}/${name}.bin
    COMMAND ${ENGINE_OCLOC} compile -q -file ${src} -device ${ENGINE_AOT_DEVICE}
            -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -I${CMAKE_SOURCE_DIR}/kernels/include -I${CMAKE_SOURCE_DIR}/shared ${opts}"
            -out_dir ${ENGINE_KERNEL_DIR} -output ${name} -output_no_suffix
    DEPENDS ${src} ${headers}
    COMMENT "ocloc ${K_SOURCE} -> ${name}.bin (${ENGINE_AOT_DEVICE})"
    VERBATIM)
  add_custom_target(kernel_${name} ALL DEPENDS ${ENGINE_KERNEL_DIR}/${name}.bin)
  set_property(GLOBAL APPEND PROPERTY ENGINE_KERNEL_TARGETS kernel_${name})
endfunction()
