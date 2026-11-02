# qwen4exp's part of the core (ADR-0024 item 3).

target_sources(engine_core PRIVATE
  dflash.cpp
  kv_share.cpp
  model.cpp
  mtp.cpp
  ngram.cpp
  prompt.cpp
  shape.cpp
  state.cpp
  swap.cpp
  vision.cpp)
# The profile that chooses the cached experts when Options::routing_profile is not given (tools/route_profile.cpp).
target_compile_definitions(engine_core PRIVATE
                           ENGINE_QWEN4EXP_ROUTING_PROFILE="${CMAKE_CURRENT_SOURCE_DIR}/routing_profile.tsv")
