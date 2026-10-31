# Kernels are OpenCL C, compiled ahead of time by ocloc into native binaries (ADR-0003), one set per chip in
# ENGINE_AOT_DEVICES, each in a directory named for the chip's IP version, which the core matches to each card's. ocloc
# writes each module's SPIR-V beside its binary; the core builds a card's modules from it when no directory matches.
# The source is built with -cl-std=CL3.0: without it ocloc compiles OpenCL C 1.2. Divides and sqrt are rounded
# correctly, as ggml's CPU code rounds them, so a quantizer's amax / 127 gives ggml's bytes (ADR-0052).

find_program(ENGINE_OCLOC ocloc)
if(NOT ENGINE_OCLOC)
  message(FATAL_ERROR "ocloc not found: install intel-ocloc, pinned in docs/adr/0002-pin-the-toolchain.md")
endif()
set(ENGINE_AOT_DEVICES "bmg-g21;bmg-g31" CACHE STRING "ocloc -device values, one binary set each (ADR-0002, ADR-0057)")
set(ENGINE_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "")
file(MAKE_DIRECTORY ${ENGINE_KERNEL_DIR})

# Each device's IP version, as `ocloc ids` gives it and Level Zero reports it: "20.1.0" for bmg-g21.
set(ENGINE_KERNEL_IPS "")
foreach(device ${ENGINE_AOT_DEVICES})
  execute_process(COMMAND ${ENGINE_OCLOC} ids ${device} OUTPUT_VARIABLE ids RESULT_VARIABLE rc)
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" ip "${ids}")
  if(NOT rc EQUAL 0 OR NOT ip)
    message(FATAL_ERROR "ocloc ids ${device} gave no IP version")
  endif()
  list(APPEND ENGINE_KERNEL_IPS ${ip})
  file(MAKE_DIRECTORY ${ENGINE_KERNEL_DIR}/${ip})
endforeach()
list(GET ENGINE_KERNEL_IPS 0 ENGINE_KERNEL_SPIRV_IP)
set(ENGINE_KERNEL_SPIRV_DIR "${ENGINE_KERNEL_DIR}/${ENGINE_KERNEL_SPIRV_IP}" CACHE INTERNAL "")

# engine_add_kernel_module(<name> SOURCE <file.cl> [OPTIONS <build options>...])
# Writes <ip>/<name>.bin and <ip>/<name>.spv for each device, and <name>.options: the -cl- options a build from the
# SPIR-V needs. A variant is another module of the same source with other -D options. Every header under kernels/ and
# shared/ is a dependency, and both directories are on the include path.
function(engine_add_kernel_module name)
  cmake_parse_arguments(K "" "SOURCE" "OPTIONS" ${ARGN})
  set(src ${CMAKE_SOURCE_DIR}/${K_SOURCE})
  file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/kernels/*.h ${CMAKE_SOURCE_DIR}/shared/*.h)
  list(JOIN K_OPTIONS " " opts)
  set(native "-cl-fp32-correctly-rounded-divide-sqrt")
  foreach(o ${K_OPTIONS})
    if(o MATCHES "^-cl-")
      string(APPEND native " ${o}")
    endif()
  endforeach()
  file(WRITE ${ENGINE_KERNEL_DIR}/${name}.options "${native}")
  set(outputs "")
  foreach(device ip IN ZIP_LISTS ENGINE_AOT_DEVICES ENGINE_KERNEL_IPS)
    set(out ${ENGINE_KERNEL_DIR}/${ip})
    add_custom_command(
      OUTPUT ${out}/${name}.bin ${out}/${name}.spv
      COMMAND ${ENGINE_OCLOC} compile -q -file ${src} -device ${device}
              -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -I${CMAKE_SOURCE_DIR}/kernels/include -I${CMAKE_SOURCE_DIR}/shared ${opts}"
              -out_dir ${out} -output ${name} -output_no_suffix
      DEPENDS ${src} ${headers}
      COMMENT "ocloc ${K_SOURCE} -> ${ip}/${name}.bin (${device})"
      VERBATIM)
    list(APPEND outputs ${out}/${name}.bin)
  endforeach()
  add_custom_target(kernel_${name} ALL DEPENDS ${outputs})
  set_property(GLOBAL APPEND PROPERTY ENGINE_KERNEL_TARGETS kernel_${name})
endfunction()
