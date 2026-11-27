# Kernels are OpenCL C, compiled ahead of time by ocloc into native binaries (ADR-0003), one set per chip in
# ENGINE_AOT_DEVICES, each in a directory named for the chip's IP version, which the core matches to each card's. Each
# module compiles to SPIR-V once per chip family, then to each chip's binary via -spirv_input (ADR-0081); the core
# builds from SPIR-V when no directory matches. The source is built with -cl-std=CL3.0: without it ocloc compiles OpenCL
# C 1.2. Divides and sqrt are rounded correctly, as ggml's CPU code rounds them (ADR-0052).

option(ENGINE_PREBUILT_KERNELS "Use kernel binaries and SPIR-V already in ENGINE_KERNEL_DIR" OFF)
if(NOT ENGINE_PREBUILT_KERNELS)
  find_program(ENGINE_OCLOC ocloc)
  if(NOT ENGINE_OCLOC)
    message(FATAL_ERROR "ocloc not found: install intel-ocloc, pinned in docs/adr/0002-pin-the-toolchain.md")
  endif()
endif()
set(ENGINE_AOT_DEVICES "bmg-g21;bmg-g31" CACHE STRING "ocloc -device values, one binary set each (ADR-0002, ADR-0057)")
set(ENGINE_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "")
file(MAKE_DIRECTORY ${ENGINE_KERNEL_DIR})

# Known device-to-IP mappings let ENGINE_PREBUILT_KERNELS configure without invoking ocloc.
set(_engine_ip_bmg-g21 "20.1.0")
set(_engine_ip_bmg-g31 "20.2.0")

# Each device's IP version, as `ocloc ids` gives it and Level Zero reports it: "20.1.0" for bmg-g21. The first device of
# each IP major holds the SPIR-V shared across chips in that family.
set(ENGINE_KERNEL_IPS "")
set(ENGINE_KERNEL_MAJORS "")
foreach(device ${ENGINE_AOT_DEVICES})
  if(ENGINE_PREBUILT_KERNELS AND DEFINED _engine_ip_${device})
    set(ip "${_engine_ip_${device}}")
  else()
    if(NOT ENGINE_OCLOC)
      find_program(ENGINE_OCLOC ocloc REQUIRED)
    endif()
    execute_process(COMMAND ${ENGINE_OCLOC} ids ${device} OUTPUT_VARIABLE ids RESULT_VARIABLE rc)
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" ip "${ids}")
    if(NOT rc EQUAL 0 OR NOT ip)
      message(FATAL_ERROR "ocloc ids ${device} gave no IP version")
    endif()
  endif()
  list(APPEND ENGINE_KERNEL_IPS ${ip})
  file(MAKE_DIRECTORY ${ENGINE_KERNEL_DIR}/${ip})
  string(REGEX MATCH "^[0-9]+" major "${ip}")
  if(NOT major IN_LIST ENGINE_KERNEL_MAJORS)
    list(APPEND ENGINE_KERNEL_MAJORS ${major})
    set(_spv_dev_${major} ${device})
    set(_spv_ip_${major} ${ip})
  endif()
endforeach()
list(GET ENGINE_KERNEL_IPS 0 ENGINE_KERNEL_SPIRV_IP)
set(ENGINE_KERNEL_SPIRV_DIR "${ENGINE_KERNEL_DIR}/${ENGINE_KERNEL_SPIRV_IP}" CACHE INTERNAL "")

if(NOT ENGINE_PREBUILT_KERNELS)
  file(GLOB_RECURSE ENGINE_KERNEL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_SOURCE_DIR}/kernels/*.h
       ${CMAKE_SOURCE_DIR}/shared/*.h
       ${CMAKE_SOURCE_DIR}/core/models/*/kernel_sizes.h
       ${CMAKE_SOURCE_DIR}/core/models/*/layout.h
       ${CMAKE_SOURCE_DIR}/core/models/*/vision_sizes.h
       ${CMAKE_SOURCE_DIR}/core/models/*/yarn.h)
endif()

# engine_add_kernel_module(<name> SOURCE <file.cl> [OPTIONS <build options>...])
# Writes <ip>/<name>.spv once per chip family, <ip>/<name>.bin for each device, and <name>.options: the -cl- options a
# build from the SPIR-V needs. A variant is another module of the same source with other -D options. Every header under
# kernels/, shared/ and the model size headers is a dependency, and kernels/include and shared are on the include path.
function(engine_add_kernel_module name)
  cmake_parse_arguments(K "" "SOURCE" "OPTIONS" ${ARGN})
  set(native "-cl-fp32-correctly-rounded-divide-sqrt")
  foreach(o ${K_OPTIONS})
    if(o MATCHES "^-cl-")
      string(APPEND native " ${o}")
    endif()
  endforeach()
  if(ENGINE_PREBUILT_KERNELS)
    if(NOT EXISTS ${ENGINE_KERNEL_DIR}/${name}.options OR NOT EXISTS ${ENGINE_KERNEL_SPIRV_DIR}/${name}.spv)
      message(FATAL_ERROR "ENGINE_PREBUILT_KERNELS=ON missing ${name}.options or SPIR-V in ${ENGINE_KERNEL_DIR}")
    endif()
    foreach(ip ${ENGINE_KERNEL_IPS})
      if(NOT EXISTS ${ENGINE_KERNEL_DIR}/${ip}/${name}.bin)
        message(FATAL_ERROR "ENGINE_PREBUILT_KERNELS=ON missing ${ENGINE_KERNEL_DIR}/${ip}/${name}.bin")
      endif()
    endforeach()
    add_custom_target(kernel_${name})
    set_property(GLOBAL APPEND PROPERTY ENGINE_KERNEL_TARGETS kernel_${name})
    return()
  endif()

  set(src ${CMAKE_SOURCE_DIR}/${K_SOURCE})
  list(JOIN K_OPTIONS " " opts)
  file(WRITE ${ENGINE_KERNEL_DIR}/${name}.options "${native}")
  foreach(major ${ENGINE_KERNEL_MAJORS})
    set(spv_out ${ENGINE_KERNEL_DIR}/${_spv_ip_${major}})
    add_custom_command(
      OUTPUT ${spv_out}/${name}.spv
      COMMAND ${ENGINE_OCLOC} compile -q -spv_only -file ${src} -device ${_spv_dev_${major}}
              -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -I${CMAKE_SOURCE_DIR}/kernels/include -I${CMAKE_SOURCE_DIR}/shared ${opts}"
              -out_dir ${spv_out} -output ${name} -output_no_suffix
      DEPENDS ${src} ${ENGINE_KERNEL_HEADERS}
      COMMENT "ocloc ${K_SOURCE} -> ${_spv_ip_${major}}/${name}.spv"
      VERBATIM)
  endforeach()
  set(outputs "")
  foreach(device ip IN ZIP_LISTS ENGINE_AOT_DEVICES ENGINE_KERNEL_IPS)
    string(REGEX MATCH "^[0-9]+" major "${ip}")
    set(spv ${ENGINE_KERNEL_DIR}/${_spv_ip_${major}}/${name}.spv)
    set(out ${ENGINE_KERNEL_DIR}/${ip})
    add_custom_command(
      OUTPUT ${out}/${name}.bin
      COMMAND ${ENGINE_OCLOC} compile -q -spirv_input -file ${spv} -device ${device}
              -options "${native}"
              -out_dir ${out} -output ${name} -output_no_suffix
      DEPENDS ${spv}
      COMMENT "ocloc ${_spv_ip_${major}}/${name}.spv -> ${ip}/${name}.bin (${device})"
      VERBATIM)
    list(APPEND outputs ${out}/${name}.bin)
  endforeach()
  add_custom_target(kernel_${name} ALL DEPENDS ${outputs})
  set_property(GLOBAL APPEND PROPERTY ENGINE_KERNEL_TARGETS kernel_${name})
endfunction()
