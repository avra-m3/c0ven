# Engine tools (ADR-0024): run by hand, not part of the engine.

add_executable(route_profile route_profile.cpp)
target_link_libraries(route_profile PRIVATE engine_core engine_warnings)
