// The kernel modules built by kernels/CMakeLists.txt, loaded on each card when first asked for.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "ze.h"

namespace engine::ze {

class Library {
 public:
  Library(Runtime &rt, std::string dir = ENGINE_KERNEL_DIR) : rt_(rt), dir_(std::move(dir)) {}
  // The kernel `name` from module `module` (the file <dir>/<module>.bin) on `card`.
  ze_kernel_handle_t get(uint32_t card, const std::string &module, const std::string &name) {
    auto &m = modules_[{card, module}];
    if (!m) m = std::make_unique<Module>(rt_, card, dir_ + "/" + module + ".bin");
    return m->kernel(name);
  }

 private:
  Runtime &rt_;
  std::string dir_;
  std::map<std::tuple<uint32_t, std::string>, std::unique_ptr<Module>> modules_;
};

}  // namespace engine::ze
