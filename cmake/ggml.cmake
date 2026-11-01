# ggml's CPU backend, at the commit ADR-0012 pins: the CPU computes the experts the cards miss with its kernels
# (ADR-0004), and the kernel tests take it as their reference (docs/TESTING.md, level 3). The image holds the tree
# at /opt/llama.cpp; ENGINE_GGML_DIR points elsewhere for a build outside it.

set(ENGINE_GGML_DIR "/opt/llama.cpp" CACHE PATH "llama.cpp tree at 3cf03257f219afbe7334045ff7c6a06ac68c627d")
if(NOT EXISTS ${ENGINE_GGML_DIR}/ggml/CMakeLists.txt)
  message(FATAL_ERROR "ggml not found in ENGINE_GGML_DIR=${ENGINE_GGML_DIR}: the image clones it (Dockerfile)")
endif()

# ggml is built once for each x86-64 level below, and each build goes into a library of its own,
# core/libengine-<level>.so; the control plane loads the first level the CPU has (engine/api/native.py). The levels
# and their flags are ggml's own (GGML_CPU_ALL_VARIANTS in its src/CMakeLists.txt), most features first. zen4 is the
# reference build's flags (docs/measurements/REFERENCE_OUTPUTS.md, "The build line"). x64 needs nothing beyond
# x86-64, so every x86-64 CPU has a level. ggml's other levels add nothing the experts' dot products use.
set(ENGINE_CPU_LEVELS_ALL zen4 icelake cascadelake skylakex alderlake haswell sandybridge sse42 x64)
set(ENGINE_CPU_LEVELS "${ENGINE_CPU_LEVELS_ALL}" CACHE STRING
    "The x86-64 levels to build ggml for, most features first; the first is the tests' and libengine.so's")
set(_level_zen4 SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
set(_level_icelake SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI)
set(_level_cascadelake SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI)
set(_level_skylakex SSE42 AVX F16C FMA AVX2 BMI2 AVX512)
set(_level_alderlake SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
set(_level_haswell SSE42 AVX F16C FMA AVX2 BMI2)
set(_level_sandybridge SSE42 AVX)
set(_level_sse42 SSE42)
set(_level_x64 "")
set(_ggml_options SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
# Each option's CPU features, in __builtin_cpu_supports's names (the core's check, core/cpu/features.h) and in
# /proc/cpuinfo's (the control plane's choice): the GCC -m flags ggml's ggml-cpu/CMakeLists.txt gives it.
set(_builtin_SSE42 sse4.2)
set(_builtin_AVX avx)
set(_builtin_F16C f16c)
set(_builtin_FMA fma)
set(_builtin_AVX2 avx2)
set(_builtin_BMI2 bmi2)
set(_builtin_AVX_VNNI avxvnni)
set(_builtin_AVX512 avx512f avx512cd avx512vl avx512dq avx512bw)
set(_builtin_AVX512_VBMI avx512vbmi)
set(_builtin_AVX512_VNNI avx512vnni)
set(_builtin_AVX512_BF16 avx512bf16)
set(_cpuinfo_SSE42 sse4_2)
set(_cpuinfo_AVX avx)
set(_cpuinfo_F16C f16c)
set(_cpuinfo_FMA fma)
set(_cpuinfo_AVX2 avx2)
set(_cpuinfo_BMI2 bmi2)
set(_cpuinfo_AVX_VNNI avx_vnni)
set(_cpuinfo_AVX512 avx512f avx512cd avx512vl avx512dq avx512bw)
set(_cpuinfo_AVX512_VBMI avx512vbmi)
set(_cpuinfo_AVX512_VNNI avx512_vnni)
set(_cpuinfo_AVX512_BF16 avx512_bf16)

# Static and position independent, so that each build links into its own shared library. No OpenMP: the engine
# calls ggml's dot products from its own threads.
include(ExternalProject)
set(_levels_text "")
foreach(level ${ENGINE_CPU_LEVELS})
  if(NOT level IN_LIST ENGINE_CPU_LEVELS_ALL)
    message(FATAL_ERROR "ENGINE_CPU_LEVELS names '${level}', not one of ${ENGINE_CPU_LEVELS_ALL}")
  endif()
  set(args)
  set(builtin)
  set(cpuinfo)
  foreach(opt ${_ggml_options})
    if(opt IN_LIST _level_${level})
      list(APPEND args -DGGML_${opt}=ON)
      list(APPEND builtin ${_builtin_${opt}})
      list(APPEND cpuinfo ${_cpuinfo_${opt}})
    else()
      list(APPEND args -DGGML_${opt}=OFF)
    endif()
  endforeach()
  set(dir ${CMAKE_BINARY_DIR}/ggml-${level})
  set(libs ${dir}/lib/libggml.a ${dir}/lib/libggml-cpu.a ${dir}/lib/libggml-base.a)
  ExternalProject_Add(ggml_build_${level}
    SOURCE_DIR ${ENGINE_GGML_DIR}/ggml
    BINARY_DIR ${dir}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${dir}/lib -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF
               -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_SYCL=OFF -DGGML_VULKAN=OFF
               -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF ${args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${dir} --target ggml ggml-cpu ggml-base
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${libs})
  add_library(ggml_${level} INTERFACE)
  add_dependencies(ggml_${level} ggml_build_${level})
  target_link_libraries(ggml_${level} INTERFACE ${libs} Threads::Threads m)
  list(JOIN builtin "," builtin)
  list(JOIN cpuinfo " " cpuinfo)
  # The features this level's ggml uses: a library or test built on it checks them before it calls ggml.
  target_compile_definitions(ggml_${level} INTERFACE ENGINE_CPU_LEVEL="${level}" ENGINE_CPU_FEATURES="${builtin}")
  string(APPEND _levels_text "${level}\t${cpuinfo}\n")
endforeach()
# The control plane reads the levels from here, next to the libraries, most features first.
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/core/cpu_levels.tsv CONTENT "${_levels_text}")

# The tests and tools take the first level.
list(GET ENGINE_CPU_LEVELS 0 ENGINE_CPU_TEST_LEVEL)
add_library(engine_ggml ALIAS ggml_${ENGINE_CPU_TEST_LEVEL})

# ggml's headers, for the core, which links no ggml of its own; ggml-common.h holds the i-quant codebooks, and the
# core reads them from there and uploads them to the cards.
set(ENGINE_GGML_INCLUDE_DIR ${ENGINE_GGML_DIR}/ggml/include CACHE INTERNAL "")
set(ENGINE_GGML_SRC_DIR ${ENGINE_GGML_DIR}/ggml/src CACHE INTERNAL "")
