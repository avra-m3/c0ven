# One level's ggml (cmake/ggml.cmake), as a subdirectory, as the engine built it before: a standalone build of the
# llama.cpp tree's ggml wants files only the ggml repository has.
cmake_minimum_required(VERSION 3.25)
project(ggml_level C CXX)
add_subdirectory(${ENGINE_GGML_DIR}/ggml ggml)
