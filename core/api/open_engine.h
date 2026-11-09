// engine_open with a model the caller makes, so a test can run the C ABI's code, its engine thread included, on a
// model of its own without a card (core/tests/engine_thread_test.cpp).
#pragma once

#include <memory>
#include <string>

#include "api/model.h"
#include "engine_api.h"

namespace engine {

using OpenModel = std::unique_ptr<Model> (*)(const std::string &path, const Options &opt);

// engine_open, the model from opener(config->model_path, the options from config) in place of open_model's. Errors
// as engine_open's.
engine_status open_engine(const engine_config *config, engine_info *info, engine_ctx **out, OpenModel opener);

}  // namespace engine
