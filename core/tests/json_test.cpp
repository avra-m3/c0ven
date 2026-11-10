// Level 1 (docs/TESTING.md): the JSON reader (util/json.h) and DFlash 2's config.json as read_df_config takes it
// (models/qwen4exp/dflash.cpp), from written files. No card.
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "models/qwen4exp/model.h"
#include "test_util.h"
#include "util/json.h"

using engine::Json;
using engine::parse_json;

namespace {

bool throws(const std::string &text) {
  try {
    parse_json(text);
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

// The config.json incoai/Qwen3.8-Flash-Next-DFlash2 publishes, trimmed to the values the core reads and a few it
// does not.
std::string config(const std::string &hidden) {
  return R"({"architectures": ["DFlash2DraftModel"], "dflash_config": {"block_size": 8, "conv_group_size": 16,
    "conv_kernel_size": 2, "mask_token_id": 248077, "selector_rank": 256, "target_layer_ids": [5, 15, 25, 35, 45]},
    "head_dim": 128, "hidden_size": )" + hidden + R"(, "intermediate_size": 7680,
    "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8, "pad_token_id": null,
    "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "sliding_window": 2048, "tie_word_embeddings": false, "use_sliding_window": true})";
}

}  // namespace

int main() {
  {
    const Json v = parse_json(R"( {"a": [1, -2.5e3, true, false, null], "b": {"c": "x\"\n\u0041"}, "d": []} )");
    EXPECT(v.at("a").array.size() == 5, "the array's values");
    EXPECT(v.at("a").array[1].number == -2500 && v.at("a").array[2].boolean && !v.at("a").array[3].boolean,
           "numbers and booleans");
    EXPECT(v.at("a").array[4].kind == Json::Kind::Null, "null");
    EXPECT(v.at("b").at("c").string == "x\"\nA", "a string's escapes");
    EXPECT(v.at("d").kind == Json::Kind::Array && v.at("d").array.empty(), "an empty array");
    EXPECT(v.find("e") == nullptr && v.at("a").find("x") == nullptr, "find of a missing member");
  }
  EXPECT(throws("{\"a\": 1,}"), "a trailing comma");
  EXPECT(throws("[1] 2"), "text after the value");
  EXPECT(throws("{\"a\": \"\\u00e9\"}"), "an escape past ASCII");
  EXPECT(throws("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["), "nesting past 64");

  const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("json_test." + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  auto write = [&](const std::string &name, const std::string &text) { std::ofstream(dir / name) << text; };
  {
    EXPECT(!engine::qwen4exp::is_dflash_dir(dir.string()), "an empty directory");
    write("config.json", config("2560"));
    EXPECT(!engine::qwen4exp::is_dflash_dir(dir.string()), "config.json without the weights");
    write("model.safetensors", "");
    EXPECT(engine::qwen4exp::is_dflash_dir(dir.string()), "both files");
    const auto c = engine::qwen4exp::read_df_config(dir.string());
    EXPECT(c.mask_token == 248077, "the mask token");
    EXPECT(c.targets[0] == 5 && c.targets[4] == 45, "the layers it reads");
  }
  {
    write("config.json", config("4096"));
    bool refused = false;
    try {
      engine::qwen4exp::read_df_config(dir.string());
    } catch (const std::runtime_error &) {
      refused = true;
    }
    EXPECT(refused, "a hidden size other than the build's");
  }
  {
    std::filesystem::remove(dir / "model.safetensors");
    write("dflash-manifest.json", "{}");
    bool refused = false;
    try {
      engine::qwen4exp::is_dflash_dir(dir.string());
    } catch (const std::runtime_error &) {
      refused = true;
    }
    EXPECT(refused, "the split layout");
  }
  std::filesystem::remove_all(dir);

  std::printf("json_test: ok\n");
  return 0;
}
