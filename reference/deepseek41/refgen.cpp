// What upstream llama.cpp computes for deepseek41 (ADR-0074 item 6), one token at a time on its CPU backend with f16
// caches, as RefModel and the engine's decode windows read a prompt. Links libllama at that commit.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"

namespace {

// The steps RefModel's probe also gives (core/models/deepseek41/reference.h), and the routes and top-k.
const std::set<std::string> kNames = {
    "hc_init",      "engram_rows",   "engram_kv",      "engram_gate", "engram_out",        "attn_norm",
    "qr_norm",      "q",             "kv",             "comp_latent", "idx_score",         "idx_top_k",
    "attn_derope",  "attn_wo_a",     "attn_out",       "hc_attn_post", "ffn_norm",         "ffn_moe_logits",
    "ffn_moe_topk", "ffn_moe_out",   "ffn_shexp",      "ffn_out",     "l_out",             "result_norm",
    "result_output"};

struct Dump {
  FILE *f = nullptr;
  uint32_t pos = 0;
  int32_t token = 0;
  bool on = false;
  std::vector<float> values;
};

// "q-12" is ("q", 12); a name with no layer is (name, -1).
void split(const char *name, std::string &base, int32_t &layer) {
  base = name;
  layer = -1;
  const size_t dash = base.rfind('-');
  if (dash != std::string::npos && dash + 1 < base.size() &&
      base.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
    layer = std::atoi(base.c_str() + dash + 1);
    base.resize(dash);
  }
}

// Writes each wanted tensor as a record of u32 magic, u32 position, i32 token, i32 layer, u32 name bytes, u32 values,
// then the name, then the values as f32. ds41_ref_model_test's compare_dump reads them.
bool on_tensor(ggml_tensor *t, bool ask, void *user) {
  auto &d = *static_cast<Dump *>(user);
  std::string base;
  int32_t layer;
  split(t->name, base, layer);
  const bool wanted = d.on && kNames.count(base) != 0;
  if (ask || !wanted) return ask ? wanted : true;
  const size_t n = (size_t)ggml_nelements(t);
  d.values.resize(n);
  if (t->type == GGML_TYPE_F32) {
    std::memcpy(d.values.data(), t->data, n * 4);
  } else if (t->type == GGML_TYPE_I32) {
    for (size_t i = 0; i < n; i++) d.values[i] = (float)((const int32_t *)t->data)[i];
  } else if (const auto to_float = ggml_get_type_traits(t->type)->to_float) {
    to_float(t->data, d.values.data(), (int64_t)n);
  } else {
    return true;
  }
  const uint32_t head[6] = {0x44343144u, d.pos, (uint32_t)d.token, (uint32_t)layer, (uint32_t)base.size(), (uint32_t)n};
  std::fwrite(head, sizeof head, 1, d.f);
  std::fwrite(base.data(), 1, base.size(), d.f);
  std::fwrite(d.values.data(), 4, n, d.f);
  return true;
}

// A prompt line is "<id> <token> <token> ...".
std::vector<llama_token> prompt_of(const char *file, const std::string &id) {
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string got;
    s >> got;
    if (got != id) continue;
    std::vector<llama_token> tokens;
    for (llama_token t; s >> t;) tokens.push_back(t);
    return tokens;
  }
  std::fprintf(stderr, "no prompt %s in %s\n", id.c_str(), file);
  std::exit(1);
}

}  // namespace

//   refgen gen  <model> <prompt line file> <id> <new tokens> <threads> >> greedy.jsonl
//   refgen dump <model> <prompt line file> <id> <new tokens> <threads> <positions> <out.bin>
// gen prints one JSON line: the id, the prompt's size, the greedy tokens and why it stopped.
int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (!((mode == "gen" && argc == 7) || (mode == "dump" && argc == 9))) {
    std::fprintf(stderr, "usage: refgen gen|dump <model> <prompts> <id> <new tokens> <threads> [<positions> <out.bin>]\n");
    return 2;
  }
  const std::string id = argv[4];
  const int n_new = std::atoi(argv[5]), threads = std::atoi(argv[6]);
  const std::vector<llama_token> prompt = prompt_of(argv[3], id);
  const uint32_t positions = mode == "dump" ? (uint32_t)std::atoi(argv[7]) : 0;

  Dump dump;
  if (mode == "dump") {
    dump.f = std::fopen(argv[8], "wb");
    if (!dump.f) {
      std::fprintf(stderr, "cannot write %s\n", argv[8]);
      return 1;
    }
  }

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;  // under a memory limit the file's pages evict only each other
  mp.use_extra_bufts = false;           // weights stay in the file's layout, no repacking into more memory
  llama_model *model = llama_model_load_from_file(argv[2], mp);
  if (!model) return 1;

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = (uint32_t)(prompt.size() + (size_t)n_new + 16);
  cp.n_batch = 1;
  cp.n_ubatch = 1;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.type_k = GGML_TYPE_F16;
  cp.type_v = GGML_TYPE_F16;
  if (dump.f) {
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &dump;
  }
  llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) return 1;

  llama_sampler *greedy = llama_sampler_init_greedy();
  const llama_vocab *vocab = llama_model_get_vocab(model);
  std::vector<llama_token> out;
  std::string stop = "length";
  uint32_t pos = 0;
  llama_token token = prompt.at(0);
  for (size_t i = 0;; i++) {
    if (i < prompt.size()) token = prompt[i];
    dump.pos = pos;
    dump.token = token;
    dump.on = pos < positions;
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch) != 0) {
      std::fprintf(stderr, "decode failed at position %u\n", pos);
      return 1;
    }
    pos++;
    if (i + 1 < prompt.size()) continue;
    if (dump.f && pos >= positions) break;
    if (out.size() >= (size_t)n_new) break;
    token = llama_sampler_sample(greedy, ctx, -1);
    out.push_back(token);
    if (llama_vocab_is_eog(vocab, token)) {
      stop = "eog";
      break;
    }
  }

  if (dump.f) {
    std::fclose(dump.f);
  } else {
    std::printf("{\"id\": \"%s\", \"n_prompt\": %zu, \"stop\": \"%s\", \"tokens\": [", id.c_str(), prompt.size(), stop.c_str());
    for (size_t i = 0; i < out.size(); i++) std::printf("%s%d", i ? ", " : "", out[i]);
    std::printf("]}\n");
  }
  llama_sampler_free(greedy);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
