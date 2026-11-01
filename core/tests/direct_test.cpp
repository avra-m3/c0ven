// Level 1 (docs/TESTING.md): which shard the loader probes, and the probe on a file shorter than its reads, on sparse
// files in a directory under the working one (ctest's is the build tree, which takes O_DIRECT). No card, no model.
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "load/direct.h"
#include "test_util.h"

using engine::DirectReader;

namespace {

// A sparse file of bytes bytes at path.
void make_file(const std::string &path, uint64_t bytes) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  EXPECT(fd >= 0, "cannot create %s", path.c_str());
  EXPECT(::ftruncate(fd, (off_t)bytes) == 0, "cannot size %s", path.c_str());
  ::close(fd);
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  std::string tmpl = (fs::current_path() / "direct_test.XXXXXX").string();
  EXPECT(::mkdtemp(tmpl.data()) != nullptr, "cannot make a temporary directory");
  const std::string dir = tmpl;
  const std::string meta = dir + "/m-00001-of-00003.gguf", a = dir + "/m-00002-of-00003.gguf",
                    b = dir + "/m-00003-of-00003.gguf";

  // A split file whose first shard holds only metadata: the largest shard is probed, the first of equal ones.
  make_file(meta, 6u << 20);
  make_file(a, DirectReader::kProbeSpan);
  make_file(b, DirectReader::kProbeSpan);
  EXPECT(DirectReader::largest({meta, a, b}) == a, "the largest of equal shards is not the first");
  make_file(b, DirectReader::kProbeSpan + 4096);
  EXPECT(DirectReader::largest({meta, a, b}) == b, "the largest shard is not chosen");
  EXPECT(DirectReader::largest({meta}) == meta, "one file is not its own largest");
  bool threw = false;
  try {
    DirectReader::largest({});
  } catch (const std::runtime_error &) {
    threw = true;
  }
  EXPECT(threw, "no files did not throw");
  threw = false;
  try {
    DirectReader::largest({meta, dir + "/missing.gguf"});
  } catch (const std::runtime_error &) {
    threw = true;
  }
  EXPECT(threw, "a missing file did not throw");

  // A file shorter than the probe's reads is not probed: the default count, a rate of 0, nothing thrown.
  for (const uint64_t bytes : {uint64_t(0), uint64_t(6u << 20), DirectReader::kProbeSpan - 4096}) {
    make_file(meta, bytes);
    double rate = -1;
    const unsigned n = DirectReader::probe_threads(meta, &rate);
    EXPECT(n == DirectReader::kThreads && rate == 0, "a file of %llu B: %u threads at %f B/s",
           (unsigned long long)bytes, n, rate);
  }

  // A file of exactly the probe's span is probed: one of its counts, at a rate above 0.
  int status = 0;
  try {
    double rate = 0;
    const unsigned n = DirectReader::probe_threads(a, &rate);
    EXPECT((n == 4 || n == 8 || n == 16) && rate > 0, "a file of the probe's span: %u threads at %f B/s", n, rate);
    std::printf("PASS: the largest shard is probed; a short file gets %u threads; a full one %u\n",
                DirectReader::kThreads, n);
  } catch (const std::runtime_error &e) {
    // A filesystem that refuses O_DIRECT (ADR-0032) cannot run the full probe; any other error fails.
    if (std::string(e.what()).find("O_DIRECT") == std::string::npos) {
      fs::remove_all(dir);
      throw;
    }
    status = test::skip(std::string("the full probe: ") + e.what());
  }
  fs::remove_all(dir);
  return status;
}
