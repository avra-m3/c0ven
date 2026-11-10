# The C++20 core: Level Zero, memory, loading, the step, the CPU's expert work (ADR-0022, ADR-0024).

add_library(engine_core STATIC
  api/draft_gate.cpp
  api/lookup.cpp
  api/swap_cadence.cpp
  api/sampling.cpp
  cpu/experts.cpp
  cpu/features.cpp
  cpu/multirow.cpp
  load/codebooks.cpp
  load/direct.cpp
  load/gguf.cpp
  load/runs.cpp
  models/open.cpp
  util/parallel.cpp
  ze/ze.cpp)
target_include_directories(engine_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/shared
                           ${CMAKE_SOURCE_DIR}/include)
target_include_directories(engine_core SYSTEM PUBLIC ${ENGINE_GGML_INCLUDE_DIR})
target_include_directories(engine_core PRIVATE ${ENGINE_GGML_SRC_DIR})
# The core links no ggml: each library below links one level's (cmake/ggml.cmake), and the tests the first level's.
target_link_libraries(engine_core PUBLIC PkgConfig::LEVEL_ZERO Threads::Threads PRIVATE engine_warnings)
target_compile_definitions(engine_core PUBLIC ENGINE_KERNEL_DIR="${ENGINE_KERNEL_DIR}"
                           ENGINE_KERNEL_SPIRV_DIR="${ENGINE_KERNEL_SPIRV_DIR}")

# Each architecture adds its own sources (ADR-0061); models/open.cpp picks one by the file's general.architecture.
add_subdirectory(models/qwen4exp)

# The multi-row dots copy ggml's AVX2 paths; multirow.cpp checks the CPU before using them.
set_source_files_properties(cpu/multirow.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")

add_subdirectory(tests)

# The libraries the control plane loads (ctypes): the C ABI over the core, one for each CPU level, libengine-<level>.so.
# libengine.so links to the first level's; the control plane loads, from beside it, the first level the CPU has
# (engine/api/native.py).
foreach(level ${ENGINE_CPU_LEVELS})
  if(level STREQUAL ENGINE_CPU_TEST_LEVEL)
    set(target engine)
  else()
    set(target engine_${level})
  endif()
  add_library(${target} SHARED api/engine_api.cpp)
  target_link_libraries(${target} PRIVATE engine_core ggml_${level} engine_warnings)
  # Only the C ABI is exported: the static libraries' symbols (the core's, ggml's) stay inside.
  target_link_options(${target} PRIVATE -Wl,--exclude-libs,ALL)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME engine-${level} C_VISIBILITY_PRESET hidden
                        CXX_VISIBILITY_PRESET hidden)
  target_compile_definitions(${target} PRIVATE ENGINE_API_EXPORT)
endforeach()
add_custom_command(TARGET engine POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink libengine-${ENGINE_CPU_TEST_LEVEL}.so libengine.so
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} VERBATIM)
