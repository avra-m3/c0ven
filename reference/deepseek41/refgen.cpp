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
#include "../src/llama-ext.h"  // the layer inputs the drafter reads, and its keep chances

namespace {

// The steps RefModel's probe also gives (core/models/deepseek41/reference.h), and the routes and top-k.
const std::set<std::string> kNames = {
    "hc_init",      "engram_rows",   "engram_kv",      "engram_gate", "engram_out",        "attn_norm",
    "qr_norm",      "q",             "kv",             "comp_latent", "idx_score",         "idx_top_k",
    "attn_derope",  "attn_wo_a",     "attn_out",       "hc_attn_post", "ffn_norm",         "ffn_moe_logits",
    "ffn_moe_topk", "ffn_moe_out",   "ffn_shexp",      "ffn_out",     "l_out",             "result_norm",
    "result_output"};

// The drafter's steps RefDrafter's probe also gives (core/models/deepseek41/drafter.h), and its routes.
const std::set<std::string> kDraftNames = {
    "main_x",       "kv_injected",    "hc_init",      "attn_norm",   "qr_norm",   "q",       "kv",
    "attn_derope",  "attn_out",       "hc_attn_post", "ffn_norm",    "ffn_moe_logits", "ffn_moe_topk",
    "ffn_moe_out",  "ffn_shexp",      "ffn_out",      "l_out",       "hc_out",    "result_norm", "result_output",
    "dspark_block_mask"};

struct Dump {
  FILE *f = nullptr;
  uint32_t pos = 0;
  int32_t token = 0;
  bool on = false;
  const std::set<std::string> *names = &kNames;
  std::vector<float> values;
};

void record(FILE *f, uint32_t pos, int32_t token, int32_t layer, const std::string &name, const float *v, size_t n) {
  const uint32_t head[6] = {0x44343144u, pos, (uint32_t)token, (uint32_t)layer, (uint32_t)name.size(), (uint32_t)n};
  std::fwrite(head, sizeof head, 1, f);
  std::fwrite(name.data(), 1, name.size(), f);
  std::fwrite(v, 4, n, f);
}

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
  const bool wanted = d.on && d.names->count(base) != 0;
  if (ask || !wanted) return ask ? wanted : true;
  const size_t n = (size_t)ggml_nelements(t);
  d.values.resize(n);
  if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) {
    // By the tensor's strides: a view of a wider tensor (the top-k of a sort) is not one run of memory.
    float *out = d.values.data();
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
      for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
        for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
          for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
            const char *at = (const char *)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
            *out++ = t->type == GGML_TYPE_F32 ? *(const float *)at : (float)*(const int32_t *)at;
          }
  } else if (const auto to_float = ggml_get_type_traits(t->type)->to_float) {
    to_float(t->data, d.values.data(), (int64_t)n);
  } else {
    return true;
  }
  record(d.f, d.pos, d.token, layer, base, d.values.data(), n);
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

int fail(const char *what) {
  std::fprintf(stderr, "%s\n", what);
  return 1;
}

// The model's greedy text with the drafter beside it, as llama.cpp's draft-dspark runs the two: after each token the
// model runs, the inputs of the layers the drafter names go to the drafter at that position (dsp_features, then its
// main_x and kv_injected); from the prompt's last token on, a block of the drafter's size is drafted from the token
// the model chose. A block's records start with dsp_block (its size; the record's token is the block's first), then
// the graph's values for <blocks> of them from block <first block> on, then dsp_tokens (the greedy chain's drafts),
// dsp_keep and, for those blocks, dsp_logits. Prints one JSON line: the tokens, each block's drafts and how many of them the
// model's own next tokens equal, from the first.
int draft_mode(char **argv) {
  const std::string id = argv[5];
  const int n_new = std::atoi(argv[6]), threads = std::atoi(argv[7]), first = std::atoi(argv[8]), blocks = std::atoi(argv[9]);
  const std::vector<llama_token> prompt = prompt_of(argv[4], id);
  Dump dump;
  dump.names = &kDraftNames;
  dump.f = std::fopen(argv[10], "wb");
  if (!dump.f) return fail("cannot write the dump");

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;
  mp.use_extra_bufts = false;
  llama_model *model = llama_model_load_from_file(argv[2], mp);
  llama_model *drafter = model ? llama_model_load_from_file(argv[3], mp) : nullptr;
  if (!drafter) return fail("a model did not load");

  char buf[32] = {};
  if (llama_model_meta_val_str(drafter, "dflash.block_size", buf, sizeof buf) < 0) return fail("the drafter names no block size");
  const int block = std::atoi(buf);
  const int32_t n_targets = (int32_t)llama_model_target_layer_ids_n(drafter);
  const int32_t *targets = llama_model_target_layer_ids(drafter);
  const int32_t n_embd = llama_model_n_embd(model), n_embd_d = llama_model_n_embd(drafter);
  const llama_vocab *vocab = llama_model_get_vocab(model);
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);
  const llama_token noise = llama_vocab_mask(llama_model_get_vocab(drafter));
  if (block < 1 || n_targets < 1 || noise < 0) return fail("the drafter's block size, layers or noise token");

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = (uint32_t)(prompt.size() + (size_t)n_new + 16 + (size_t)block);
  cp.n_batch = (uint32_t)block;
  cp.n_ubatch = (uint32_t)block;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.type_k = GGML_TYPE_F16;
  cp.type_v = GGML_TYPE_F16;
  llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) return fail("no context for the model");
  for (int32_t k = 0; k < n_targets; k++) llama_set_embeddings_layer_inp(ctx, (uint32_t)targets[k], true);

  cp.ctx_other = ctx;
  cp.cb_eval = on_tensor;
  cp.cb_eval_user_data = &dump;
  llama_context *dctx = llama_init_from_model(drafter, cp);
  if (!dctx) return fail("no context for the drafter");
  llama_set_embeddings_nextn(dctx, true, true);
  llama_set_causal_attn(dctx, false);

  llama_batch feed = llama_batch_init(1, n_targets * n_embd, 1);
  llama_batch noise_block = llama_batch_init(block, 0, 1);
  llama_sampler *greedy = llama_sampler_init_greedy();
  std::vector<llama_token> out;
  std::vector<std::vector<llama_token>> drafts;
  std::vector<std::vector<float>> keeps;
  std::vector<float> logits((size_t)block * (size_t)n_vocab), keep((size_t)block), chosen((size_t)block);
  std::string stop = "length";
  uint32_t pos = 0;
  llama_token token = prompt.at(0);
  for (size_t i = 0;; i++) {
    if (i < prompt.size()) token = prompt[i];
    llama_batch one = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, one) != 0) return fail("the model's decode failed");

    for (int32_t k = 0; k < n_targets; k++) {
      const float *in = llama_get_embeddings_layer_inp(ctx, (uint32_t)targets[k]);
      if (!in) return fail("a layer's input was not kept");
      std::memcpy(feed.embd + (size_t)k * (size_t)n_embd, in, (size_t)n_embd * sizeof(float));
    }
    record(dump.f, pos, token, -1, "dsp_features", feed.embd, (size_t)n_targets * (size_t)n_embd);
    feed.n_tokens = 1;
    feed.pos[0] = (llama_pos)pos;
    feed.n_seq_id[0] = 1;
    feed.seq_id[0][0] = 0;
    feed.logits[0] = 0;
    dump.pos = pos;
    dump.token = token;
    dump.on = true;
    if (llama_decode(dctx, feed) != 0) return fail("the drafter's feed failed");
    pos++;
    if (i + 1 < prompt.size()) continue;
    if (out.size() >= (size_t)n_new) break;
    token = llama_sampler_sample(greedy, ctx, -1);
    out.push_back(token);
    if (llama_vocab_is_eog(vocab, token)) {
      stop = "eog";
      break;
    }

    const bool whole = drafts.size() >= (size_t)first && drafts.size() < (size_t)first + (size_t)blocks;
    const float size = (float)block;
    record(dump.f, pos, token, -1, "dsp_block", &size, 1);
    noise_block.n_tokens = block;
    for (int32_t b = 0; b < block; b++) {
      noise_block.token[b] = b == 0 ? token : noise;
      noise_block.pos[b] = (llama_pos)pos + b;
      noise_block.n_seq_id[b] = 1;
      noise_block.seq_id[b][0] = 0;
      noise_block.logits[b] = 1;
    }
    dump.pos = pos;
    dump.token = token;
    dump.on = whole;
    if (llama_decode(dctx, noise_block) != 0) return fail("the drafter's block failed");
    const float *conf = llama_get_embeddings_nextn(dctx);
    if (!conf) return fail("the drafter gave no keep chances");
    drafts.emplace_back();
    for (int32_t b = 0; b < block; b++) {
      const float *row = llama_get_logits_ith(dctx, b);
      std::memcpy(logits.data() + (size_t)b * (size_t)n_vocab, row, (size_t)n_vocab * sizeof(float));
      const llama_token best = (llama_token)(std::max_element(row, row + n_vocab) - row);
      drafts.back().push_back(best);
      chosen[(size_t)b] = (float)best;
      keep[(size_t)b] = conf[(size_t)b * (size_t)n_embd_d];
    }
    keeps.push_back(keep);
    record(dump.f, pos, token, -1, "dsp_tokens", chosen.data(), chosen.size());
    record(dump.f, pos, token, -1, "dsp_keep", keep.data(), keep.size());
    if (whole) record(dump.f, pos, token, -1, "dsp_logits", logits.data(), logits.size());
    // The block's own keys leave: the next feed is at the block's first position.
    if (!llama_memory_seq_rm(llama_get_memory(dctx), 0, (llama_pos)pos, -1)) return fail("the drafter's block stayed in its cache");
  }
  std::fclose(dump.f);

  std::printf("{\"id\": \"%s\", \"n_prompt\": %zu, \"stop\": \"%s\", \"tokens\": [", id.c_str(), prompt.size(), stop.c_str());
  for (size_t i = 0; i < out.size(); i++) std::printf("%s%d", i ? ", " : "", out[i]);
  std::printf("], \"blocks\": [");
  for (size_t b = 0; b < drafts.size(); b++) {
    // Block b starts from out[b]; its draft j stands for out[b + 1 + j].
    int same = 0;
    while (same < block && b + 1 + (size_t)same < out.size() && drafts[b][(size_t)same] == out[b + 1 + (size_t)same]) same++;
    std::printf("%s{\"drafts\": [", b ? ", " : "");
    for (int j = 0; j < block; j++) std::printf("%s%d", j ? ", " : "", drafts[b][(size_t)j]);
    std::printf("], \"keep\": [");
    for (int j = 0; j < block; j++) std::printf("%s%.4f", j ? ", " : "", keeps[b][(size_t)j]);
    std::printf("], \"same\": %d}", same);
  }
  std::printf("]}\n");
  llama_sampler_free(greedy);
  llama_batch_free(feed);
  llama_batch_free(noise_block);
  llama_free(dctx);
  llama_free(ctx);
  llama_model_free(drafter);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}

}  // namespace

//   refgen draft <model> <drafter> <prompt line file> <id> <new tokens> <threads> <first block> <blocks> <out.bin> >> drafts.jsonl
//   refgen gen  <model> <prompt line file> <id> <new tokens> <threads> >> greedy.jsonl
//   refgen dump <model> <prompt line file> <id> <new tokens> <threads> <positions> <out.bin>
// gen prints one JSON line: the id, the prompt's size, the greedy tokens and why it stopped.
int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "draft" && argc == 11) return draft_mode(argv);
  if (!((mode == "gen" && argc == 7) || (mode == "dump" && argc == 9))) {
    std::fprintf(stderr, "usage: refgen gen|dump <model> <prompts> <id> <new tokens> <threads> [<positions> <out.bin>]\n"
                         "       refgen draft <model> <drafter> <prompts> <id> <new tokens> <threads> <first block> <blocks> <out.bin>\n");
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
