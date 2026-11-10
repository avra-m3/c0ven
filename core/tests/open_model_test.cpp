// Level 1 (docs/TESTING.md): open_model's choice of architecture (api/model.h), on small GGUF headers written here. No
// card and no model file.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/model.h"
#include "load/gguf.h"
#include "test_util.h"

namespace {

struct Writer {
  std::vector<uint8_t> b;
  template <class T>
  void put(T v) {
    const auto *p = reinterpret_cast<const uint8_t *>(&v);
    b.insert(b.end(), p, p + sizeof v);
  }
  void str(const std::string &s) {
    put<uint64_t>(s.size());
    b.insert(b.end(), s.begin(), s.end());
  }
};

// A GGUF version 3 file of no tensors: a number and an array of strings, then general.architecture when arch is
// not empty.
void write_gguf(const std::string &path, const std::string &arch) {
  Writer w;
  w.put<uint32_t>(0x46554747u);
  w.put<uint32_t>(3);
  w.put<uint64_t>(0);
  w.put<uint64_t>(arch.empty() ? 2 : 3);
  w.str("general.alignment");
  w.put<uint32_t>(4);  // u32
  w.put<uint32_t>(32);
  w.str("tokenizer.ggml.tokens");
  w.put<uint32_t>(9);  // array
  w.put<uint32_t>(8);  // of strings
  w.put<uint64_t>(2);
  w.str("a");
  w.str("bc");
  if (!arch.empty()) {
    w.str("general.architecture");
    w.put<uint32_t>(8);
    w.str(arch);
  }
  FILE *f = std::fopen(path.c_str(), "wb");
  EXPECT(f && std::fwrite(w.b.data(), 1, w.b.size(), f) == w.b.size(), "cannot write %s", path.c_str());
  std::fclose(f);
}

// The message open_model throws, or "" when it does not throw.
std::string open_error(const std::string &path) {
  try {
    engine::open_model(path, engine::Options{});
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

}  // namespace

int main() {
  char dir[] = "/tmp/open_model_test.XXXXXX";
  EXPECT(mkdtemp(dir), "mkdtemp");
  const std::string d = dir;

  // The first shard alone gives the architecture: the others are never opened.
  const std::string ds = d + "/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf";
  write_gguf(ds, "deepseek41");
  EXPECT(engine::gguf_architecture(ds) == "deepseek41", "architecture of the first shard");
  const std::string e = open_error(ds);
  EXPECT(!e.empty() && e.find("does not run") == std::string::npos, "deepseek41 reaches its model: '%s'", e.c_str());

  const std::string other = d + "/other.gguf";
  write_gguf(other, "llama");
  const std::string refused = open_error(other);
  EXPECT(refused.find("does not run architecture llama") != std::string::npos,
         "an architecture the engine does not run is named: '%s'", refused.c_str());

  const std::string qw = d + "/one.gguf";
  write_gguf(qw, "qwen4exp");
  EXPECT(engine::gguf_architecture(qw) == "qwen4exp", "architecture after other keys");

  const std::string none = d + "/none.gguf";
  write_gguf(none, "");
  EXPECT(open_error(none).find("no general.architecture") != std::string::npos, "a file without the key");
  EXPECT(open_error(d + "/missing.gguf").find("cannot open") != std::string::npos, "a missing file");

  for (const std::string &p : {ds, other, qw, none}) std::remove(p.c_str());
  rmdir(dir);
  std::printf("open_model_test: ok\n");
  return 0;
}
