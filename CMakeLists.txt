cmake_minimum_required(VERSION 3.25)
# The project's one version is in VERSION (ADR-0042).
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION C0VEN_VERSION LIMIT_COUNT 1)
if(NOT C0VEN_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "VERSION holds '${C0VEN_VERSION}', not MAJOR.MINOR.PATCH")
endif()
project(engine VERSION ${C0VEN_VERSION} LANGUAGES C CXX)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS VERSION)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ENGINE_WERROR "Treat warnings as errors" OFF)
add_library(engine_warnings INTERFACE)
# Level Zero descriptors are initialised as {STRUCTURE_TYPE}, so -Wmissing-field-initializers is off.
target_compile_options(engine_warnings INTERFACE -Wall -Wextra -Wno-missing-field-initializers
                       $<$<BOOL:${ENGINE_WERROR}>:-Werror>)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LEVEL_ZERO IMPORTED_TARGET level-zero)
if(NOT LEVEL_ZERO_FOUND)
  message(FATAL_ERROR "Level Zero not found: install libze-dev (and libze1), pinned in docs/adr/0002-pin-the-toolchain.md")
endif()

include(cmake/kernels.cmake)
include(cmake/ggml.cmake)

enable_testing()

add_subdirectory(kernels)
add_subdirectory(core)
add_subdirectory(tools)
