# ggml's CPU backend, at the commit ADR-0012 pins: the CPU computes the experts the cards miss with its kernels
# (ADR-0004), and the kernel tests take it as their reference (docs/TESTING.md, level 3). The image holds the tree
# at /opt/llama.cpp; ENGINE_GGML_DIR points elsewhere for a build outside it.

set(ENGINE_GGML_DIR "/opt/llama.cpp" CACHE PATH "llama.cpp tree at 3cf03257f219afbe7334045ff7c6a06ac68c627d")
if(NOT EXISTS ${ENGINE_GGML_DIR}/ggml/CMakeLists.txt)
  message(FATAL_ERROR "ggml not found in ENGINE_GGML_DIR=${ENGINE_GGML_DIR}: the image clones it (Dockerfile)")
endif()

# The CPU flags of the reference build (docs/measurements/REFERENCE_OUTPUTS.md, "The build line"), static and position
# independent so that it links into the engine's shared library. No OpenMP: the engine calls ggml's dot products
# from its own threads.
set(BUILD_SHARED_LIBS OFF)
foreach(opt GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
            GGML_AVX512_BF16)
  set(${opt} ON CACHE BOOL "" FORCE)
endforeach()
foreach(opt GGML_NATIVE GGML_OPENMP GGML_BUILD_TESTS GGML_BUILD_EXAMPLES GGML_SYCL GGML_VULKAN GGML_CUDA GGML_METAL
            GGML_BLAS GGML_LLAMAFILE)
  set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
# The features those flags compile ggml for (its ggml-cpu/CMakeLists.txt, GCC's -m flags), in
# __builtin_cpu_supports's names: the core refuses to open on a CPU without one (core/cpu/features.h).
set(ENGINE_CPU_FEATURES "avx,avx2,fma,f16c,bmi2,avx512f,avx512cd,avx512vl,avx512dq,avx512bw,avx512vbmi,avx512vnni,avx512bf16"
    CACHE INTERNAL "")
add_subdirectory(${ENGINE_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL SYSTEM)

# ggml-common.h holds the i-quant codebooks; the core reads them from there and uploads them to the cards.
set(ENGINE_GGML_SRC_DIR ${ENGINE_GGML_DIR}/ggml/src CACHE INTERNAL "")
