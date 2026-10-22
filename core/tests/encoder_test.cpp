// Level 4 (docs/TESTING.md): the image encoder (core/models/qwen4exp/vision.cpp) against reference rows for the same
// images and video frame pairs. Gate, per image: the rows' mean cosine similarity at least 0.999 and the least at
// least 0.95; no largest-error gate (docs/measurements/MEASUREMENTS.md#2026-10-17-encoder-reference says why).
// encoder_test [reference dir, default $FLASH_REF_DIR/vision] [cards, default 2]; the encoder's file is
// FLASH_MODEL_DIR's mmproj-Qwen3.8-Flash-Next-BF16.gguf. FLASH_ENCODER_OUT=<dir> writes the engine's rows there.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

#include "models/qwen4exp/model.h"
#include "test_util.h"

using namespace engine;

namespace {

std::vector<char> slurp(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

}  // namespace

int main(int argc, char **argv) {
  if (test::gpus() == 0) return test::no_gpu();
  const std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  const char *rd = std::getenv("FLASH_REF_DIR");
  const std::string dir = argc > 1 ? argv[1] : std::string(rd ? rd : "/scratch/probes") + "/vision";
  const char *md = std::getenv("FLASH_MODEL_DIR");
  const std::string vision = std::string(md ? md : "/models") + "/mmproj-Qwen3.8-Flash-Next-BF16.gguf";
  if (!std::ifstream(vision)) return test::skip("no " + vision);
  std::vector<std::string> names;
  for (const char *n : {"img64", "img252", "img4096", "vid64", "vid252"})
    if (std::ifstream(dir + "/" + n + ".emb")) names.push_back(n);
  if (names.empty()) return test::skip("no reference embeddings in " + dir);

  qwen4exp::Options opt;
  opt.max_cards = argc > 2 ? (uint32_t)std::stoul(argv[2]) : 2;
  opt.max_ctx = 8192;
  opt.vision_path = vision;
  qwen4exp::Model model(path, opt);
  std::printf("loaded on %u card(s); images up to %u tokens\n", model.info().cards, model.image_tokens_max());
  int failures = 0;
  for (const std::string &name : names) {
    // .grid: height and width in patches, then the frames (absent: 1); .u8: patches in the encoder's order; .emb:
    // MTMD_DEBUG_EMBEDDINGS' format.
    uint32_t gh = 0, gw = 0, frames = 1;
    {
      std::ifstream g(dir + "/" + name + ".grid");
      g >> gh >> gw;
      if (uint32_t f; g >> f) frames = f;
    }
    const std::vector<char> pix = slurp(dir + "/" + name + ".u8"), emb = slurp(dir + "/" + name + ".emb");
    const uint32_t tokens = gh / 2 * (gw / 2);
    EXPECT(pix.size() == (size_t)gh * gw * 768 * frames, "%s: %zu bytes of pixels", name.c_str(), pix.size());
    int32_t hdr[2];
    std::memcpy(hdr, emb.data(), 8);
    EXPECT((uint32_t)hdr[0] == tokens && hdr[1] == 2560, "%s: the reference has %d x %d", name.c_str(), hdr[0], hdr[1]);
    const float *ref = reinterpret_cast<const float *>(emb.data() + 8);
    if (tokens > model.image_tokens_max()) {
      std::printf("%-8s %u tokens: over the limit, skipped\n", name.c_str(), tokens);
      continue;
    }
    qwen4exp::Image im;
    im.grid_h = gh;
    im.grid_w = gw;
    im.frames = frames;
    im.pixels = reinterpret_cast<const uint8_t *>(pix.data());
    std::vector<float> out((size_t)tokens * 2560);
    for (int run = 0; run < 2; run++) {
      const auto t0 = std::chrono::steady_clock::now();
      model.encode_image(im, out.data());
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (run == 1) std::printf("%-8s %u tokens: %.1f ms (second run)\n", name.c_str(), tokens, ms);
    }
    if (const char *od = std::getenv("FLASH_ENCODER_OUT")) {  // <dir>/<name>.emb, the reference's format
      std::ofstream f(std::string(od) + "/" + name + ".emb", std::ios::binary);
      const int32_t h[2] = {(int32_t)tokens, 2560};
      f.write(reinterpret_cast<const char *>(h), 8);
      f.write(reinterpret_cast<const char *>(out.data()), (std::streamsize)(out.size() * 4));
    }
    double maxerr = 0, maxref = 0, mincos = 1, sumcos = 0;
    for (uint32_t t = 0; t < tokens; t++) {
      double dot = 0, na = 0, nb = 0;
      for (uint32_t i = 0; i < 2560; i++) {
        const double r = ref[(size_t)t * 2560 + i], g = out[(size_t)t * 2560 + i];
        maxref = std::max(maxref, std::fabs(r));
        const double e = std::fabs(r - g);
        maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
        dot += r * g;
        na += r * r;
        nb += g * g;
      }
      const double cs = dot / std::sqrt(na * nb);
      mincos = std::isnan(cs) ? -1 : std::min(mincos, cs);
      sumcos += cs;
    }
    const double rel = maxerr / maxref, meancos = sumcos / tokens;
    const bool ok = meancos >= 0.999 && mincos >= 0.95;
    failures += !ok;
    std::printf("%-8s cosine mean %.6f, least %.6f; max error %.2e of %.3g %s\n", name.c_str(), meancos, mincos, rel,
                maxref, ok ? "ok" : "FAIL");
  }
  EXPECT(failures == 0, "%d image(s) over tolerance", failures);
  std::printf("PASS\n");
  return 0;
}
