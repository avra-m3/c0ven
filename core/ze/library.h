// The kernel modules built by kernels/CMakeLists.txt, loaded on each card when first asked for: the binaries built for
// the card's IP version, or, for a chip with none (or ENGINE_KERNEL_TARGET=spirv), the SPIR-V compiled at load.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ze.h"

namespace engine::ze {

class Library {
 public:
  Library(Runtime &rt, std::string dir = ENGINE_KERNEL_DIR, std::string spirv_dir = ENGINE_KERNEL_SPIRV_DIR);
  // The kernel `name` from module `module` on `card`.
  ze_kernel_handle_t get(uint32_t card, const std::string &module, const std::string &name);
  // The card's modules come from SPIR-V compiled at load.
  bool spirv(uint32_t card) const { return spirv_.at(card); }

 private:
  Runtime &rt_;
  std::string dir_, spirv_dir_;
  std::vector<bool> spirv_;
  std::map<std::tuple<uint32_t, std::string>, std::unique_ptr<Module>> modules_;
};

}  // namespace engine::ze
