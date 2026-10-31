# The C++20 core: Level Zero, memory, loading, the step, the CPU's expert work (ADR-0022, ADR-0024).

add_library(engine_core STATIC
  api/lookup.cpp
  api/swap_cadence.cpp
  api/sampling.cpp
  cpu/experts.cpp
  cpu/multirow.cpp
  load/codebooks.cpp
  load/direct.cpp
  load/gguf.cpp
  load/runs.cpp
  models/qwen4exp/dflash.cpp
  models/qwen4exp/kv_share.cpp
  models/qwen4exp/model.cpp
  models/qwen4exp/ngram.cpp
  models/qwen4exp/mtp.cpp
  models/qwen4exp/prompt.cpp
  models/qwen4exp/state.cpp
  models/qwen4exp/swap.cpp
  models/qwen4exp/vision.cpp
  util/parallel.cpp
  ze/ze.cpp)
target_include_directories(engine_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/shared
                           ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/models/qwen4exp)
target_include_directories(engine_core PRIVATE ${ENGINE_GGML_SRC_DIR})
target_link_libraries(engine_core PUBLIC PkgConfig::LEVEL_ZERO Threads::Threads ggml PRIVATE engine_warnings)
target_compile_definitions(engine_core PUBLIC ENGINE_KERNEL_DIR="${ENGINE_KERNEL_DIR}"
                           ENGINE_ROUTING_PROFILE="${CMAKE_CURRENT_SOURCE_DIR}/models/qwen4exp/routing_profile.tsv")

# The multi-row dots copy ggml's AVX2 paths; multirow.cpp checks the CPU before using them.
set_source_files_properties(cpu/multirow.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")

add_subdirectory(tests)

# The library the control plane loads (ctypes): the C ABI over the core.
add_library(engine SHARED api/engine_api.cpp)
target_link_libraries(engine PRIVATE engine_core engine_warnings)
# Only the C ABI is exported: the static libraries' symbols (the core's, ggml's) stay inside.
target_link_options(engine PRIVATE -Wl,--exclude-libs,ALL)
set_target_properties(engine PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(engine PRIVATE ENGINE_API_EXPORT)
