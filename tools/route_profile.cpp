// route_profile <reference dir> <out.tsv> [max tokens per prompt]: reads every reference prompt with its reference
// output through the prompt path and writes how many times each layer routed to each expert, the profile the engine
// chooses its cached experts by (engine_config.routing_profile).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "models/qwen4exp/model.h"
#include "../core/tests/test_util.h"

using namespace engine;

namespace {
std::vector<int32_t> ints_after(const std::string &line, const std::string &key) {
  std::vector<int32_t> v;
  size_t p = line.find("\"" + key + "\":[");
  if (p == std::string::npos) return v;
  p = line.find('[', p) + 1;
  while (line[p] != ']') {
    size_t q;
    v.push_back(std::stoi(line.substr(p), &q));
    p += q;
    if (line[p] == ',') p++;
  }
  return v;
}
std::string str_after(const std::string &line, const std::string &key) {
  size_t p = line.find("\"" + key + "\":\"");
  if (p == std::string::npos) return "";
  p += key.size() + 4;
  return line.substr(p, line.find('"', p) - p);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: route_profile <reference dir> <out.tsv> [max tokens per prompt]\n");
    return 2;
  }
  std::string dir = argv[1];
  size_t cap = argc > 3 ? std::stoul(argv[3]) : 32768;
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  std::vector<std::pair<std::string, std::vector<int32_t>>> seqs;
  {
    std::ifstream pf(dir + "/prompts.jsonl"), gf(dir + "/llamacpp-3cf0325-IQ3_S-greedy.jsonl");
    std::string line;
    while (std::getline(pf, line)) seqs.push_back({str_after(line, "id"), ints_after(line, "tokens")});
    while (std::getline(gf, line)) {
      std::string id = str_after(line, "id");
      for (auto &s : seqs)
        if (s.first == id) {
          auto out = ints_after(line, "tokens");
          s.second.insert(s.second.end(), out.begin(), out.end());
        }
    }
  }
  qwen4exp::Options opt;
  opt.max_ctx = 32768 + 1024;
  opt.profile_routing = true;
  qwen4exp::Model model(path, opt);
  uint64_t total = 0;
  for (auto &[id, seq] : seqs) {
    size_t n = std::min(seq.size(), cap);
    model.reset_slot(0);
    for (size_t at = 0; at < n;) {
      uint32_t len = (uint32_t)std::min<size_t>(model.prompt_chunk(), n - at);
      model.prompt(0, (uint32_t)at, seq.data() + at, len, at ? seq[at - 1] : qwen4exp::kNoToken,
                   at > 1 ? seq[at - 2] : qwen4exp::kNoToken);
      at += len;
    }
    total += n;
    std::printf("%-16s %6zu tokens\n", id.c_str(), n);
  }
  auto counts = model.routing_counts();
  FILE *f = std::fopen(argv[2], "w");
  EXPECT(f, "cannot write %s", argv[2]);
  std::fprintf(f, "# routing profile: %llu tokens of the reference prompts and outputs (%s), read by the prompt path\n",
               (unsigned long long)total, dir.c_str());
  std::fprintf(f, "# layer expert rows\n");
  for (uint32_t l = 0; l < 48; l++)
    for (uint32_t e = 0; e < 512; e++) std::fprintf(f, "%u %u %u\n", l, e, counts[(size_t)l * 512 + e]);
  std::fclose(f);
  std::printf("wrote %s\n", argv[2]);
  return 0;
}
