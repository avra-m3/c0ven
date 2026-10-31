#include "direct.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace engine {

namespace {

// pread until the range is in or the file ends; returns the bytes read.
uint64_t pread_all(int fd, uint8_t *dst, uint64_t bytes, uint64_t offset, const std::string &path) {
  uint64_t got = 0;
  while (got < bytes) {
    ssize_t n = ::pread(fd, dst + got, bytes - got, (off_t)(offset + got));
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) throw std::runtime_error("reading " + path + ": " + std::strerror(errno));
    if (n == 0) break;  // the end of the file, inside the last aligned block
    got += (uint64_t)n;
  }
  return got;
}

}  // namespace

void AlignedFree::operator()(uint8_t *p) const { std::free(p); }

AlignedBuffer aligned_buffer(uint64_t bytes) {
  void *p = nullptr;
  bytes = (bytes + DirectReader::kAlign - 1) / DirectReader::kAlign * DirectReader::kAlign;
  if (::posix_memalign(&p, DirectReader::kAlign, bytes ? bytes : DirectReader::kAlign) != 0)
    throw std::runtime_error("cannot allocate a staging buffer of " + std::to_string(bytes) + " B");
  return AlignedBuffer(static_cast<uint8_t *>(p));
}

DirectReader::DirectReader(const std::vector<std::string> &paths, unsigned threads)
    : paths_(paths), pool_(threads > 0 ? threads - 1 : 0) {
  for (const std::string &p : paths) {
    int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) {
      for (int f : fds_) ::close(f);
      throw std::runtime_error("cannot open " + p + " with O_DIRECT (ADR-0032): " +
                               std::strerror(errno));
    }
    fds_.push_back(fd);
  }
}

DirectReader::~DirectReader() {
  for (int fd : fds_) ::close(fd);
}

unsigned DirectReader::probe_threads(const std::string &path, double *rate) {
  constexpr uint64_t kProbe = 256ull << 20;
  const unsigned counts[] = {4, 8, 16};
  auto buf = aligned_buffer(kProbe);
  unsigned best = 8;
  double best_rate = 0;
  for (unsigned i = 0; i < 3; i++) {
    DirectReader r({path}, counts[i]);
    const auto t0 = std::chrono::steady_clock::now();
    r.read(0, (1 + i) * kProbe, kProbe, buf.get());
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (kProbe / s > best_rate) best_rate = kProbe / s, best = counts[i];
  }
  if (rate) *rate = best_rate;
  return best;
}

uint64_t DirectReader::span(uint64_t offset, uint64_t bytes) {
  const uint64_t a = offset / kAlign * kAlign, b = (offset + bytes + kAlign - 1) / kAlign * kAlign;
  return b - a;
}

const uint8_t *DirectReader::read_one(uint32_t file, uint64_t offset, uint64_t bytes, uint8_t *buf) const {
  const uint64_t a = offset / kAlign * kAlign, n = span(offset, bytes);
  const uint64_t got = pread_all(fds_.at(file), buf, n, a, paths_.at(file));
  if (got < offset + bytes - a) throw std::runtime_error(paths_.at(file) + " is shorter than its tensor table says");
  bytes_read_ += n;
  return buf + (offset - a);
}

const uint8_t *DirectReader::read(uint32_t file, uint64_t offset, uint64_t bytes, uint8_t *buf) {
  const uint64_t a = offset / kAlign * kAlign, n = span(offset, bytes);
  const uint64_t pieces = (n + kPiece - 1) / kPiece;
  std::atomic<uint64_t> short_pieces{0};
  pool_.run(pieces, [&](uint64_t i) {
    const uint64_t at = i * kPiece, len = std::min(kPiece, n - at);
    const uint64_t got = pread_all(fds_.at(file), buf + at, len, a + at, paths_.at(file));
    // Only the file's last block may come back short, and only past the range.
    if (got < len && a + at + got < offset + bytes) short_pieces++;
  });
  if (short_pieces) throw std::runtime_error(paths_.at(file) + " is shorter than its tensor table says");
  bytes_read_ += n;
  return buf + (offset - a);
}

}  // namespace engine
