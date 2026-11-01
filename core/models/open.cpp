// open_model (api/model.h): the one place that names each architecture the core runs (ADR-0061).
#include <memory>
#include <stdexcept>
#include <string>

#include "api/model.h"
#include "load/gguf.h"
#include "models/qwen4exp/model.h"

namespace engine {

std::unique_ptr<Model> open_model(const std::string &path, const Options &opt) {
  const std::string arch = gguf_architecture(path);
  if (arch == "qwen4exp") return std::make_unique<qwen4exp::Model>(path, opt);
  throw std::runtime_error(path + ": the engine does not run architecture " + arch);
}

}  // namespace engine
