// Level 1 (docs/TESTING.md): deepseek41's experts off the cards, on small generated shards of real-shaped experts in a
// directory under the working one (ctest's is the build tree, which takes O_DIRECT). The arena's layout, the reads, the
// bits and pool jobs of a layer's sums, the ring at the widest step on the real file's offsets (the table given as the
// argument), the refusal of a filesystem without O_DIRECT, reads kept off the engine thread, the bounds on the host's
// waits, and the deadline thread with a fake event. No card, no model. The sums skip (77) on a CPU without the tests'
// ggml build's features.
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cpu/experts.h"
#include "cpu/features.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "load/direct.h"
#include "models/deepseek41/tiers.h"
#include "test_util.h"

using namespace engine;
using namespace engine::deepseek41;

namespace {

constexpr uint32_t kExperts = 8;  // a generated layer's, in place of 384

std::mt19937 rng(41);
std::normal_distribution<float> normal(0.0f, 1.0f);

template <typename F>
bool throws_with(const F &f, const std::string &text) {
  try {
    f();
  } catch (const std::exception &e) {
    return std::string(e.what()).find(text) != std::string::npos;
  }
  return false;
}

// A byte source of whole random ggml blocks of a type.
std::vector<uint8_t> blocks_of(ggml_type t) {
  const int bv = (int)ggml_blck_size(t), n = 256;
  std::vector<float> values((size_t)n * bv), importance(bv, 1.0f);
  for (float &v : values) v = normal(rng) * 0.02f;
  std::vector<uint8_t> out((size_t)n * ggml_type_size(t));
  ggml_quantize_init(t);
  ggml_quantize_chunk(t, values.data(), out.data(), 0, n, bv, importance.data());
  return out;
}

// Appends experts matrices of random blocks of a type to a shard after an unaligned gap; returns the first's place.
MatrixPlace add_matrix(std::vector<uint8_t> &shard, uint32_t file, ggml_type t, uint64_t expert_bytes) {
  static std::map<int, std::vector<uint8_t>> sources;
  if (!sources.count(t)) sources[t] = blocks_of(t);
  const std::vector<uint8_t> &src = sources[t];
  const size_t block = ggml_type_size(t);
  const uint64_t gap = 1000 + rng() % 3000, offset = shard.size() + gap, bytes = kExperts * expert_bytes;
  shard.resize(offset + bytes);
  for (uint64_t at = 0; at < bytes; at += block)
    std::memcpy(shard.data() + offset + at, src.data() + (rng() % (src.size() / block)) * block, block);
  return {file, offset};
}

LayerExperts add_layer(std::vector<std::vector<uint8_t>> &shards, uint32_t file, ggml_type gate, ggml_type down) {
  LayerExperts l;
  l.shape = {(uint32_t)gate, (uint32_t)down, D41_EMBD, D41_FF, D41_SWIGLU_LIMIT};
  l.gate_bytes = ggml_row_size(gate, D41_EMBD) * D41_FF;
  l.down_bytes = ggml_row_size(down, D41_FF) * D41_EMBD;
  l.experts = kExperts;
  l.down = add_matrix(shards[file], file, down, l.down_bytes);
  l.gate = add_matrix(shards[file], file, gate, l.gate_bytes);
  l.up = add_matrix(shards[file], file, gate, l.gate_bytes);
  return l;
}

// The generated file: three layers of the table's kinds in two shards, each shard ending with its last matrix, so reads
// meet the short last block.
struct Generated {
  std::filesystem::path dir;
  std::vector<std::string> paths;
  std::vector<std::vector<uint8_t>> shards;
  std::vector<LayerExperts> layers;

  explicit Generated(const std::filesystem::path &d) : dir(d), shards(2) {
    layers.push_back(add_layer(shards, 0, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S));
    layers.push_back(add_layer(shards, 1, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S));
    layers.push_back(add_layer(shards, 1, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_M));
    for (size_t i = 0; i < shards.size(); i++) {
      paths.push_back((dir / ("shard-" + std::to_string(i) + ".gguf")).string());
      std::ofstream(paths[i], std::ios::binary).write(reinterpret_cast<const char *>(shards[i].data()), (std::streamsize)shards[i].size());
    }
  }
  const uint8_t *at(const MatrixPlace &m, uint64_t bytes, uint32_t id) const {
    return shards[m.file].data() + m.offset + id * bytes;
  }
  // The expert in the file's layout, whole down.
  ExpertWeights whole(uint32_t layer, uint32_t id) const {
    const LayerExperts &l = layers[layer];
    ExpertWeights w;
    w.gate = at(l.gate, l.gate_bytes, id);
    w.up = at(l.up, l.gate_bytes, id);
    w.down = at(l.down, l.down_bytes, id);
    return w;
  }
  uint64_t largest() const {
    uint64_t b = 0;
    for (const LayerExperts &l : layers) b = std::max(b, l.expert_bytes());
    return b;
  }
};

void expect_same(const ExpertWeights &got, const ExpertWeights &want, const LayerExperts &l, const char *what) {
  EXPECT(!std::memcmp(got.gate, want.gate, l.gate_bytes), "%s: gate differs from the file", what);
  EXPECT(!std::memcmp(got.up, want.up, l.gate_bytes), "%s: up differs from the file", what);
  EXPECT(!std::memcmp(got.down, want.down, l.down_bytes), "%s: down differs from the file", what);
}

// Every 4 KiB page of [p, p + bytes) is resident.
bool resident(const uint8_t *p, uint64_t bytes) {
  const uint64_t page = 4096, a = reinterpret_cast<uintptr_t>(p) / page * page;
  const uint64_t n = (reinterpret_cast<uintptr_t>(p) + bytes - a + page - 1) / page;
  std::vector<unsigned char> v(n);
  if (mincore(reinterpret_cast<void *>(a), n * page, v.data()) != 0) return false;
  return std::all_of(v.begin(), v.end(), [](unsigned char c) { return c & 1; });
}

void check_arena(const Generated &g, DirectReader &reader) {
  // Small pieces, so experts meet a piece's end. Each layer is in part held.
  const uint64_t largest = g.largest();
  ExpertArena arena(4 * largest, 2 * (largest / kHugeBytes + 1) * kHugeBytes);
  const std::vector<uint32_t> first = {5, 0, 2}, second = {1, 3, 4, 6, 7};
  EXPECT(arena.fill(reader, 0, g.layers[0], first) == 3, "the arena's budget holds three of layer 0");
  // One more fits, the next does not: the rest stay on the SSD.
  EXPECT(arena.fill(reader, 1, g.layers[1], second) == 1, "the budget ends the fill without a refusal");
  EXPECT(arena.pieces().size() >= 2, "experts spread over pieces");

  auto check_one = [&](uint32_t layer, uint32_t id) {
    const LayerExperts &l = g.layers[layer];
    const ExpertWeights w = arena.weights(layer, id), file = g.whole(layer, id);
    EXPECT(w.gate && w.down_parts == 2 && w.down_blocks[0] == 5 && w.down_blocks[1] == 4, "weights of an arena expert");
    EXPECT(w.up == w.gate + l.gate_bytes && w.down == w.up + l.gate_bytes, "gate, up, then down");
    EXPECT(!std::memcmp(w.gate, file.gate, l.gate_bytes), "arena gate differs from the file");
    EXPECT(!std::memcmp(w.up, file.up, l.gate_bytes), "arena up differs from the file");
    const size_t block = ggml_type_size((ggml_type)l.shape.down_type), a = 5 * block, b = 4 * block;
    for (uint32_t r = 0; r < D41_EMBD; r++) {
      const uint8_t *row = file.down + r * (a + b);
      EXPECT(!std::memcmp(w.down + r * a, row, a), "arena down part A differs at row %u", r);
      EXPECT(!std::memcmp(w.down + D41_EMBD * a + r * b, row + a, b), "arena down part B differs at row %u", r);
    }
    bool inside = false;
    for (const AnonMemory &p : arena.pieces())
      inside = inside || (w.gate >= p.data() && w.down + l.down_bytes <= p.data() + p.bytes());
    EXPECT(inside, "an expert crosses a piece");
    EXPECT(reinterpret_cast<uintptr_t>(arena.pieces()[0].data()) % kHugeBytes == 0, "a piece is 2 MiB aligned");
  };
  for (uint32_t id : first) check_one(0, id);
  check_one(1, second[0]);
  EXPECT(!arena.weights(1, second[1]).gate, "an expert past the budget has no arena weights");
  EXPECT(!arena.weights(2, 0).gate && !arena.weights(0, 1).gate, "experts not asked for have no arena weights");

  const ArenaReport rep = arena.settle();
  EXPECT(rep.experts == 4 && rep.pieces == arena.pieces().size(), "the report's counts");
  EXPECT(rep.ranges > 0 && rep.huge_ranges <= rep.ranges, "the report's huge-page coverage");
  EXPECT(rep.touched_pages >= rep.bytes / 4096, "every used page is touched: %llu pages", (unsigned long long)rep.touched_pages);
  const LayerExperts &l0 = g.layers[0];
  EXPECT(resident(arena.weights(0, 5).gate, l0.expert_bytes()), "an arena expert is resident after settle");
  std::printf("arena: %llu of %llu ranges in huge pages, %s\n", (unsigned long long)rep.huge_ranges,
              (unsigned long long)rep.ranges, rep.lock_errno ? "unlocked" : "locked");
}

// A piece from the host's reserved huge pages holds experts as any other; with none free it is the process's own.
void check_reserved(const Generated &g, DirectReader &reader) {
  ExpertArena arena(2 * g.largest(), kArenaPieceBytes, true);
  EXPECT(arena.fill(reader, 0, g.layers[0], {3, 1}) == 2, "a reserved arena holds two experts");
  const LayerExperts &l = g.layers[0];
  for (uint32_t id : {3u, 1u}) {
    const ExpertWeights w = arena.weights(0, id), file = g.whole(0, id);
    EXPECT(w.gate && !std::memcmp(w.gate, file.gate, l.gate_bytes) && !std::memcmp(w.up, file.up, l.gate_bytes),
           "a reserved arena's expert %u differs from the file", id);
  }
  const ArenaReport rep = arena.settle();
  const bool reserved = arena.pieces()[0].reserved();
  EXPECT(rep.reserved_pieces == (reserved ? 1u : 0u) && rep.pieces == 1, "the report's reserved pieces");
  if (reserved) EXPECT(rep.huge_ranges == rep.ranges, "a reserved piece is all huge pages");
  std::printf("reserved huge pages: %s\n", reserved ? "a piece took them" : "none free, the piece is the process's own memory");
}

void check_memlock() {
  // A memory lock too small refuses the arena, and the switch opens it. A process allowed to lock any amount is not
  // limited, so the refusal cannot run there, and the output says so.
  rlimit old;
  EXPECT(getrlimit(RLIMIT_MEMLOCK, &old) == 0, "getrlimit");
  rlimit low = old;
  low.rlim_cur = 1 << 16;
  EXPECT(setrlimit(RLIMIT_MEMLOCK, &low) == 0, "setrlimit");
  const std::string unlocked_before = std::getenv("ENGINE_UNLOCKED_ARENA") ? "1" : "";
  unsetenv("ENGINE_UNLOCKED_ARENA");
  bool refused = false;
  try {
    AnonMemory m(kHugeBytes);
  } catch (const std::runtime_error &e) {
    refused = true;
    EXPECT(std::string(e.what()).find("mlock2") != std::string::npos, "the refusal names mlock2: %s", e.what());
  }
  std::printf("memory lock: %s\n", refused ? "the refusal ran" : "this process may lock any amount, the refusal did not run");
  setenv("ENGINE_UNLOCKED_ARENA", "1", 1);
  {
    AnonMemory m(kHugeBytes);
    EXPECT(refused == (m.lock_errno() != 0), "the switch keeps the reason");
    std::memset(m.data(), 1, m.bytes());
  }
  if (unlocked_before.empty()) unsetenv("ENGINE_UNLOCKED_ARENA");
  EXPECT(setrlimit(RLIMIT_MEMLOCK, &old) == 0, "setrlimit");
}

// The file's table. Rows: name, type, ggml_type, shape, bytes, shard, data_offset, category.
std::vector<LayerExperts> read_table(const std::string &path) {
  std::ifstream f(path);
  EXPECT(f.good(), "cannot read %s", path.c_str());
  std::vector<LayerExperts> layers(D41_LAYERS);
  std::string line;
  uint32_t found = 0;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    std::vector<std::string> c(8);
    for (std::string &v : c) std::getline(s, v, '\t');
    if (c[7] != "experts") continue;
    const size_t dot = c[0].find('.', 4);
    const uint32_t layer = (uint32_t)std::stoul(c[0].substr(4, dot - 4));
    const std::string kind = c[0].substr(dot + 1);
    const uint64_t bytes = std::stoull(c[4]) / D41_EXPERTS;
    const MatrixPlace place{(uint32_t)std::stoul(c[5]), std::stoull(c[6])};
    LayerExperts &l = layers.at(layer);
    if (kind.rfind("ffn_gate", 0) == 0) l.gate = place, l.gate_bytes = bytes, l.shape.gate_type = (uint32_t)std::stoul(c[2]);
    if (kind.rfind("ffn_up", 0) == 0) l.up = place;
    if (kind.rfind("ffn_down", 0) == 0) l.down = place, l.down_bytes = bytes, l.shape.down_type = (uint32_t)std::stoul(c[2]);
    found++;
  }
  EXPECT(found == 3 * D41_LAYERS, "the table holds %u expert tensors", found);
  return layers;
}

uint64_t expert_span(const LayerExperts &l, uint32_t id) {
  return DirectReader::span(l.gate.offset + id * l.gate_bytes, l.gate_bytes) +
         DirectReader::span(l.up.offset + id * l.gate_bytes, l.gate_bytes) +
         DirectReader::span(l.down.offset + id * l.down_bytes, l.down_bytes);
}

// The experts of the layer whose reads take the most room.
std::vector<uint32_t> widest_ids(const LayerExperts &l, uint32_t n) {
  std::vector<uint32_t> ids(l.experts);
  for (uint32_t i = 0; i < l.experts; i++) ids[i] = i;
  std::stable_sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return expert_span(l, a) > expert_span(l, b); });
  ids.resize(n);
  return ids;
}

void check_widest_ring(const std::string &table) {
  const std::vector<LayerExperts> layers = read_table(table);
  uint64_t largest = 0;
  for (const LayerExperts &l : layers) largest = std::max(largest, l.expert_bytes());
  EXPECT(largest == 8663040, "the largest expert is %llu B", (unsigned long long)largest);
  // engine/models/deepseek41/memory.py's figure at the widest step.
  const uint64_t capacity = staging_ring_bytes(largest, STEP_MAX_ROWS);
  EXPECT(capacity == 2ull * 48 * (8663040 + 3 * 4096), "the ring's bytes are %llu", (unsigned long long)capacity);

  // Every pair of layers in flight at once, each with its 48 widest reads; a step's last layer and the next's first too.
  for (uint32_t l0 = 0; l0 < D41_LAYERS; l0++) {
    const uint32_t l1 = (l0 + 1) % D41_LAYERS;
    RingLayout ring(capacity);
    std::vector<RingRead> reads;
    for (uint32_t side = 0; side < 2; side++) {
      const LayerExperts &l = layers[side ? l1 : l0];
      for (uint32_t id : widest_ids(l, 48)) {
        const RingExpert e = ring.place(side, l, id);
        const uint64_t off[3] = {l.gate.offset + id * l.gate_bytes, l.up.offset + id * l.gate_bytes,
                                 l.down.offset + id * l.down_bytes};
        const uint64_t bytes[3] = {l.gate_bytes, l.gate_bytes, l.down_bytes};
        const RingRead *m[3] = {&e.gate, &e.up, &e.down};
        for (int i = 0; i < 3; i++) {
          EXPECT(m[i]->span == DirectReader::span(off[i], bytes[i]), "a read's span");
          EXPECT(m[i]->at % 4096 == 0, "a read's place is not 4 KiB aligned");
          reads.push_back(*m[i]);
        }
      }
    }
    std::sort(reads.begin(), reads.end(), [](const RingRead &a, const RingRead &b) { return a.at < b.at; });
    for (size_t i = 0; i < reads.size(); i++) {
      EXPECT(reads[i].at + reads[i].span <= capacity, "a read passes the ring's end");
      EXPECT(i == 0 || reads[i - 1].at + reads[i - 1].span <= reads[i].at, "reads overlap at layers %u and %u", l0, l1);
    }
  }

  // A round the ring cannot hold is refused before it overlaps anything, and a released side is free again.
  const LayerExperts &l = layers[0];
  RingLayout small(2 * expert_span(l, 0) + expert_span(l, 0) / 2);
  small.place(0, l, 0);
  small.place(1, l, 0);
  EXPECT(throws_with([&] { small.place(0, l, 0); }, "staging ring"), "a third expert must not fit");
  small.release(1);
  small.place(0, l, 0);
}

void check_reads(const Generated &g, uint64_t ring_bytes) {
  // Two live rounds, in a ring exactly as long as their reads, keep their bytes: one overlapping would change them.
  const std::vector<uint32_t> ids0 = {3, 0, 7, 1, 6}, ids1 = {2, 7, 5, 0};
  uint64_t exact = 0;
  for (uint32_t id : ids0) exact += expert_span(g.layers[0], id);
  for (uint32_t id : ids1) exact += expert_span(g.layers[2], id);
  EXPECT(exact <= ring_bytes, "the test's ring");
  SsdReader reader(g.paths, exact);
  std::vector<ExpertWeights> w0, w1;
  reader.start(0, g.layers[0], ids0, w0);
  reader.wait();
  reader.start(1, g.layers[2], ids1, w1);
  reader.wait();
  EXPECT(reader.rounds() == 2 && reader.pool_runs() == 2, "a round is one run of the reader's pool");
  for (size_t i = 0; i < ids0.size(); i++) expect_same(w0[i], g.whole(0, ids0[i]), g.layers[0], "side 0");
  for (size_t i = 0; i < ids1.size(); i++) expect_same(w1[i], g.whole(2, ids1[i]), g.layers[2], "side 1");
  EXPECT(reader.direct().bytes_read() == exact, "the reader read %llu B, not the rounds' %llu B",
         (unsigned long long)reader.direct().bytes_read(), (unsigned long long)exact);
  EXPECT(w0[0].down_parts == 1 && w0[0].down_blocks[0] == 0, "an SSD expert's down stays whole");

  // A round past the ring is refused and starts no read.
  const uint64_t before = reader.direct().bytes_read();
  std::vector<ExpertWeights> w;
  EXPECT(throws_with([&] { reader.start(1, g.layers[1], {0, 1, 2, 3, 4, 5, 6, 7}, w); }, "staging ring"), "ring refusal");
  reader.wait();
  EXPECT(reader.direct().bytes_read() == before && reader.rounds() == 2, "a refused round reads nothing");

  // A filesystem without O_DIRECT (procfs) is refused, ADR-0015 item 5.
  EXPECT(throws_with([&] { SsdReader bad({g.paths[0], "/proc/version"}, kHugeBytes); }, "O_DIRECT"),
         "a file without O_DIRECT must be refused");
}

// A step's records, with rows at or past real_rows holding routes that must be ignored.
struct Records {
  std::vector<D41MissJob> jobs;
  explicit Records(uint32_t rows) : jobs(rows) {
    for (D41MissJob &j : jobs) {
      j.nmiss = 0;
      for (float &v : j.x) v = normal(rng);
    }
  }
  void route(uint32_t row, const std::vector<std::pair<int32_t, float>> &r) {
    D41MissJob &j = jobs[row];
    j.nmiss = (uint32_t)r.size();
    for (size_t k = 0; k < r.size(); k++) j.ids[k] = r[k].first, j.w[k] = r[k].second;
  }
};

// y from compute_many on whole-down weights of the routings the predicate picks, in routing order.
std::vector<float> whole_sum(const Generated &g, uint32_t layer, const Records &rec, uint32_t real,
                             const std::function<bool(uint32_t)> &pick) {
  Pool pool(3);
  CpuExperts cpu(pool);
  std::vector<CpuExperts::Job> jobs;
  for (uint32_t r = 0; r < real; r++)
    for (uint32_t k = 0; k < rec.jobs[r].nmiss; k++)
      if (pick((uint32_t)rec.jobs[r].ids[k])) jobs.push_back({g.whole(layer, (uint32_t)rec.jobs[r].ids[k]), r, rec.jobs[r].w[k]});
  std::vector<float> y((size_t)real * D41_EMBD);
  cpu.compute_many(g.layers[layer].shape, jobs, real, rec.jobs[0].x, sizeof(D41MissJob) / sizeof(float), y.data(), D41_EMBD);
  return y;
}

constexpr uint32_t kRows = 4, kReal = 3;
constexpr uint64_t kStride = D41_EMBD + 32;

Records routed_records() {
  Records rec(kRows);
  rec.route(0, {{4, 0.5f}, {0, 0.25f}, {5, 0.125f}, {1, 0.0625f}, {3, 0.03f}, {6, 0.02f}});
  rec.route(1, {{4, 1.0f}, {2, 0.5f}});  // expert 4 again: read once
  rec.route(2, {});
  rec.route(3, {{7, 9.0f}, {7, 9.0f}, {7, 9.0f}, {7, 9.0f}, {7, 9.0f}, {7, 9.0f}});  // padded: routes to nothing
  std::fill(std::begin(rec.jobs[3].x), std::end(rec.jobs[3].x), std::nanf(""));
  return rec;
}

void check_answer(const std::vector<float> &answer, const std::vector<float> &want, const char *what) {
  for (uint32_t r = 0; r < kReal; r++)
    EXPECT(!std::memcmp(answer.data() + r * kStride, want.data() + (size_t)r * D41_EMBD, D41_EMBD * sizeof(float)),
           "%s: row %u's answer differs from the whole-down sums", what, r);
  for (uint32_t i = 0; i < D41_EMBD; i++)
    EXPECT(answer[kReal * kStride + i] == 0.0f, "%s: a padded row's answer is not zero", what);
}

void check_sums(const Generated &g) {
  SsdReader reader(g.paths, staging_ring_bytes(g.largest(), 1));
  const auto in_arena = [](uint32_t id) { return id < 3; };
  const std::vector<uint32_t> arena_ids = {0, 1, 2};
  ExpertArena arena(16 * g.largest()), none(0), all(3 * kExperts * g.largest());
  for (uint32_t l = 0; l < g.layers.size(); l++) {
    arena.fill(reader.direct(), l, g.layers[l], arena_ids);
    all.fill(reader.direct(), l, g.layers[l], {0, 1, 2, 3, 4, 5, 6, 7});
  }
  arena.settle();

  for (uint32_t layer = 0; layer < g.layers.size(); layer++) {
    const Records rec = routed_records();
    const uint32_t n_flag = 7;
    const std::vector<float> a = whole_sum(g, layer, rec, kReal, in_arena);
    const std::vector<float> s = whole_sum(g, layer, rec, kReal, [&](uint32_t id) { return !in_arena(id); });
    const std::vector<float> one = whole_sum(g, layer, rec, kReal, [](uint32_t) { return true; });
    std::vector<float> mixed(a.size());
    for (size_t i = 0; i < a.size(); i++) mixed[i] = a[i] + s[i];

    // The same bits on every run, at any number of workers.
    for (unsigned workers : {0u, 3u, 7u}) {
      Pool pool(workers);
      MissJobs mj(g.layers, arena, reader, pool);
      for (int run = 0; run < 2; run++) {
        std::vector<float> answer((size_t)kRows * kStride, 1.0f);
        uint32_t flag = 0;
        const uint64_t reads0 = reader.pool_runs(), bytes0 = reader.direct().bytes_read(), pool0 = pool.runs_started();
        MissAsk ask{layer, kRows, kReal, n_flag, rec.jobs.data(), answer.data(), kStride, &flag};
        const MissCounts c = mj.run_layer(ask);
        EXPECT(flag == n_flag, "the answer's flag holds the step");
        check_answer(answer, mixed, "arena then SSD");
        EXPECT(c.routings == 8 && c.arena_jobs == 3 && c.ssd_jobs == 5 && c.ssd_experts == 4, "the layer's counts");
        EXPECT(c.read_rounds == 1 && reader.pool_runs() == reads0 + 1, "one read round a layer, one run of the reader's pool");
        EXPECT(c.cpu_pool_runs == 2 && pool.runs_started() == pool0 + 2, "one CPU pool job a tier: %u", c.cpu_pool_runs);
        // Expert 7 of the padded row is not read, and expert 4 of two rows once.
        uint64_t want_bytes = 0;
        for (uint32_t id : {4u, 5u, 3u, 6u}) want_bytes += expert_span(g.layers[layer], id);
        EXPECT(reader.direct().bytes_read() - bytes0 == want_bytes, "the round reads its experts once each");
      }
    }
    Pool pool(7);
    {
      // One tier: one pool job, and the bits of the whole-down sum in routing order. From the arena through unequal
      // parts; from the SSD through whole downs.
      for (const ExpertArena *tier : {&all, &none}) {
        MissJobs mj(g.layers, *tier, reader, pool);
        std::vector<float> answer((size_t)kRows * kStride);
        const uint64_t reads0 = reader.pool_runs();
        const MissCounts c = mj.run_layer({layer, kRows, kReal, 1, rec.jobs.data(), answer.data(), kStride, nullptr});
        check_answer(answer, one, tier == &all ? "all arena" : "all SSD");
        EXPECT(c.cpu_pool_runs == 1, "one tier takes one CPU pool job, not %u", c.cpu_pool_runs);
        EXPECT(c.read_rounds == (tier == &all ? 0u : 1u), "reads only for the SSD tier");
        EXPECT(reader.pool_runs() == reads0 + c.read_rounds, "the reader's pool stays asleep without an SSD miss");
      }
    }
    {
      // No miss on a real row, and only a padded row routing: no job at all, no answer, no flag.
      Records quiet(kRows);
      quiet.route(3, {{7, 1.0f}});
      MissJobs mj(g.layers, arena, reader, pool);
      std::vector<float> answer((size_t)kRows * kStride, 3.0f);
      uint32_t flag = 0;
      const uint64_t pool0 = pool.runs_started(), reads0 = reader.pool_runs();
      const MissCounts c = mj.run_layer({layer, kRows, kReal, 5, quiet.jobs.data(), answer.data(), kStride, &flag});
      EXPECT(c.routings == 0 && c.cpu_pool_runs == 0 && c.read_rounds == 0, "a layer with no miss is skipped");
      EXPECT(pool.runs_started() == pool0 && reader.pool_runs() == reads0, "a skipped layer wakes no worker");
      EXPECT(flag == 0 && answer[0] == 3.0f, "a skipped layer writes no answer and no flag");
    }
  }

  // A bad record fails the job without raising the flag.
  {
    Pool pool(3);
    MissJobs mj(g.layers, arena, reader, pool);
    Records rec(1);
    rec.route(0, {{(int32_t)kExperts, 1.0f}});
    std::vector<float> answer(D41_EMBD);
    uint32_t flag = 0;
    EXPECT(throws_with([&] { mj.run_layer({0, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &flag}); }, "names expert"),
           "an id past the layer must fail");
    rec.jobs[0].nmiss = D41_USED + 1;
    EXPECT(throws_with([&] { mj.run_layer({0, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &flag}); }, "holds"),
           "too many misses must fail");
    EXPECT(flag == 0, "a failed job raises no flag");
  }
}

// A read that fails ends the job with an error and no flag, and frees its side of the ring: the next round on that side
// runs in a ring that holds only one round.
void check_read_failure(const Generated &g) {
  std::vector<LayerExperts> layers = g.layers;
  layers[1].down.offset = g.shards[1].size() + (1 << 20);  // past the file's end
  layers.push_back(g.layers[0]);                           // layer 3 reads well, on layer 1's side
  uint64_t failing = 0, next = 0;
  for (uint32_t id = 2; id < kExperts; id++) failing += expert_span(layers[1], id), next += expert_span(layers[3], id);
  SsdReader reader(g.paths, std::max(failing, next));
  ExpertArena none(0);
  Pool pool(3);
  MissJobs mj(layers, none, reader, pool);
  Records rec(1);
  rec.route(0, {{2, 1.0f}, {3, 1.0f}, {4, 1.0f}, {5, 1.0f}, {6, 1.0f}, {7, 1.0f}});
  std::vector<float> answer(D41_EMBD);
  uint32_t flag = 0;
  mj.submit({1, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &flag});
  EXPECT(throws_with([&] { mj.wait(); }, "shorter"), "a failed read reaches wait");
  EXPECT(flag == 0, "a failed read raises no flag");
  mj.submit({3, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &flag});
  mj.wait();
  EXPECT(flag == 9, "the next round on the side runs");
}

// An arena sum that fails with reads running ends the job only once they have landed, and frees their side of the ring.
void check_arena_failure(const Generated &g) {
  const uint32_t layer = 1;
  const LayerExperts &l = g.layers[layer];
  const std::vector<uint32_t> on_ssd = {3, 4, 5, 6, 7};
  uint64_t round = 0;
  for (uint32_t id : on_ssd) round += expert_span(l, id);
  SsdReader reader(g.paths, round);
  ExpertArena arena(16 * g.largest());
  arena.fill(reader.direct(), layer, l, {0, 1, 2});
  Pool pool(3);
  MissJobs mj(g.layers, arena, reader, pool);

  std::mutex mu;
  std::condition_variable cv;
  bool failed = false;
  reader.set_read_hook([&] {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::seconds(60), [&] { return failed; });
  });
  mj.set_arena_hook([&] {
    {
      std::lock_guard<std::mutex> lk(mu);
      failed = true;
      cv.notify_all();
    }
    throw std::runtime_error("the arena's sum failed");
  });

  Records rec(1);
  rec.route(0, {{0, 1.0f}, {3, 1.0f}, {4, 1.0f}, {5, 1.0f}, {6, 1.0f}, {7, 1.0f}});
  std::vector<float> answer(D41_EMBD);
  uint32_t flag = 0;
  const uint64_t before = reader.direct().bytes_read();
  mj.submit({layer, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &flag});
  EXPECT(throws_with([&] { mj.wait(); }, "the arena's sum failed"), "a failed sum reaches wait");
  EXPECT(flag == 0, "a failed sum raises no flag");
  EXPECT(reader.direct().bytes_read() - before == round, "the job ended before its reads landed");
  mj.set_arena_hook({});
  mj.submit({layer, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &flag});
  mj.wait();
  EXPECT(flag == 9, "the next round on the side runs");
}

// The engine thread hands a job over and goes on while its reads are held. A read has started before the arena's sum
// starts and is held until the sum ends, so a job that runs the two in turn fails here.
void check_engine_thread(const Generated &g) {
  SsdReader reader(g.paths, staging_ring_bytes(g.largest(), 1));
  ExpertArena arena(16 * g.largest());
  arena.fill(reader.direct(), 0, g.layers[0], {0, 1, 2});
  Pool pool(3);
  MissJobs mj(g.layers, arena, reader, pool);

  std::mutex mu;
  std::condition_variable cv;
  bool submitted = false, arena_done = false, timed_out = false, read_before_sum = false;
  std::set<std::thread::id> readers;
  reader.set_read_hook([&] {
    std::unique_lock<std::mutex> lk(mu);
    readers.insert(std::this_thread::get_id());
    cv.notify_all();
    if (!cv.wait_for(lk, std::chrono::seconds(60), [&] { return submitted && arena_done; })) timed_out = true;
  });
  mj.set_arena_start_hook([&] {
    std::unique_lock<std::mutex> lk(mu);
    // Under MissJobs::wait's bound, so a job that runs the two in turn fails on the check below.
    read_before_sum = cv.wait_for(lk, std::chrono::seconds(5), [&] { return !readers.empty(); });
  });
  mj.set_arena_hook([&] {
    std::lock_guard<std::mutex> lk(mu);
    arena_done = true;
    cv.notify_all();
  });

  Records rec(2);
  rec.route(0, {{0, 1.0f}, {4, 1.0f}});
  rec.route(1, {{1, 1.0f}, {5, 1.0f}});
  std::vector<float> answer(2 * D41_EMBD);
  uint32_t flag = 0;
  const uint64_t read_before = reader.direct().bytes_read();
  mj.submit({0, 2, 2, 3, rec.jobs.data(), answer.data(), D41_EMBD, &flag});
  // The reads are held until submitted is set below, so nothing can have landed yet.
  EXPECT(reader.direct().bytes_read() == read_before, "submit waited on a read");
  EXPECT(std::atomic_ref<uint32_t>(flag).load() == 0, "the answer came before its reads");
  {
    std::lock_guard<std::mutex> lk(mu);
    submitted = true;
    cv.notify_all();
  }
  mj.wait();
  EXPECT(read_before_sum, "no read had started when the arena's sum started");
  EXPECT(!timed_out, "the reads were still held 60 s after they started");
  EXPECT(flag == 3, "the job finished");
  EXPECT(!readers.empty() && !readers.count(std::this_thread::get_id()), "a read ran on the engine thread");
}

// Calls a bounded wait until it stops expiring: a short limit can expire on a loaded host once the reads are let go.
template <typename F>
void until_it_returns(const F &wait) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (std::chrono::steady_clock::now() < end) {
    try {
      wait();
      return;
    } catch (const std::runtime_error &e) {
      EXPECT(std::string(e.what()).find("did not") != std::string::npos || std::string(e.what()).find("still running") != std::string::npos,
             "a bounded wait failed: %s", e.what());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT(false, "a bounded wait never returned");
}

// A read that does not return fails the wait on it within the bound. Nothing reuses the ring until the read lands.
void check_wait_bounds(const Generated &g) {
  const auto limit = std::chrono::milliseconds(100);
  const std::vector<uint32_t> ids = {1, 2};
  uint64_t round = 0;
  for (uint32_t id : ids) round += expert_span(g.layers[0], id);
  std::mutex mu;
  std::condition_variable cv;
  bool go = false;
  const auto hold = [&] {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::seconds(60), [&] { return go; });
  };
  const auto let_go = [&](bool to) {
    std::lock_guard<std::mutex> lk(mu);
    go = to;
    cv.notify_all();
  };
  ExpertArena none(0);
  Pool pool(3);
  Records rec(1);
  rec.route(0, {{1, 1.0f}, {2, 1.0f}});
  std::vector<float> answer(D41_EMBD);

  {
    // SsdReader::wait, in a ring of one round, so the next round runs only on a side that was freed.
    SsdReader reader(g.paths, round, DirectReader::kThreads, limit);
    reader.set_read_hook(hold);
    std::vector<ExpertWeights> w;
    reader.start(0, g.layers[0], ids, w);
    EXPECT(throws_with([&] { reader.wait(); }, "did not land"), "a held read must fail the wait");
    EXPECT(throws_with([&] { reader.start(0, g.layers[0], ids, w); }, "still running"), "a running round refuses the next");
    reader.release(0);
    let_go(true);
    until_it_returns([&] { reader.wait(); });
    reader.start(0, g.layers[0], ids, w);
    until_it_returns([&] { reader.wait(); });
    for (size_t i = 0; i < ids.size(); i++) expect_same(w[i], g.whole(0, ids[i]), g.layers[0], "after a wait gave up");
  }
  let_go(false);
  {
    // The job's own wait on its reads: the job fails, and the next round runs once the first has landed.
    SsdReader reader(g.paths, round, DirectReader::kThreads, limit);
    reader.set_read_hook(hold);
    MissJobs mj(g.layers, none, reader, pool);
    uint32_t flag = 0;
    const MissAsk ask{0, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &flag};
    EXPECT(throws_with([&] { mj.run_layer(ask); }, "did not land"), "a held read must fail the job");
    EXPECT(flag == 0, "a job past its bound raises no flag");
    let_go(true);
    until_it_returns([&] { mj.run_layer(ask); });
    EXPECT(flag == 9, "the next round runs");
  }
  let_go(false);
  {
    // MissJobs::wait: it names the layer, drops the job not begun, and leaves the job running to finish.
    SsdReader reader(g.paths, staging_ring_bytes(g.largest(), 1));
    reader.set_read_hook(hold);
    MissJobs mj(g.layers, none, reader, pool, limit);
    uint32_t running = 0, dropped = 0;
    mj.submit({0, 1, 1, 9, rec.jobs.data(), answer.data(), D41_EMBD, &running});
    std::vector<float> other(D41_EMBD);
    mj.submit({2, 1, 1, 9, rec.jobs.data(), other.data(), D41_EMBD, &dropped});
    EXPECT(throws_with([&] { mj.wait(); }, "layer 0's miss job did not finish"), "a held job must fail the wait");
    let_go(true);
    until_it_returns([&] { mj.wait(); });
    EXPECT(std::atomic_ref<uint32_t>(running).load() == 9, "the running job finished");
    EXPECT(dropped == 0 && reader.rounds() == 1, "a job not begun is dropped");
  }
}

// A fake event: the list arrives when the test says, and signal records whether the stuck flag was already set.
class FakeEvent : public DeadlineEvent {
 public:
  explicit FakeEvent(uint32_t *stuck) : stuck_(stuck) {}
  std::atomic<bool> at{false};
  std::atomic<int> signals{0};
  std::atomic<bool> stuck_before{false};
  std::chrono::steady_clock::time_point when;
  bool reached() override { return at; }
  void signal() override {
    stuck_before = std::atomic_ref<uint32_t>(*stuck_).load() != 0;
    when = std::chrono::steady_clock::now();
    signals++;
  }

 private:
  uint32_t *stuck_;
};

void check_deadlines() {
  using clock = std::chrono::steady_clock;
  HostDeadlines d;
  uint32_t s_late = 0, s_answered = 0, s_never = 0;
  FakeEvent late(&s_late), answered(&s_answered), never(&s_never);
  const uint64_t h_late = d.hold(late, &s_late), h_answered = d.hold(answered, &s_answered), h_never = d.hold(never, &s_never);

  // The list reaches two events; the host answers one in time. The clock starts at the list's arrival, not at the hold.
  answered.at = true;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT(!d.release(h_answered), "an event the host answered in time");
  const auto reached = clock::now();
  late.at = true;

  std::this_thread::sleep_for(std::chrono::milliseconds(900));
  EXPECT(late.signals == 0 && s_late == 0, "the deadline fired before 1 s");
  for (int i = 0; i < 400 && late.signals == 0; i++) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT(late.signals == 1, "the deadline did not fire");
  const double after = std::chrono::duration<double>(late.when - reached).count();
  EXPECT(after >= 1.0 && after < 2.0, "the deadline fired %.3f s after the list arrived", after);
  EXPECT(late.stuck_before, "the stuck flag comes before the signal");
  EXPECT(s_late == 1, "the stuck flag is set");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT(late.signals == 1, "a deadline fires once");
  EXPECT(d.release(h_late), "release says the deadline passed");

  EXPECT(answered.signals == 0 && s_answered == 0, "an answered event is left alone");
  EXPECT(never.signals == 0 && s_never == 0, "an event no list reached is left alone");
  EXPECT(!d.release(h_never), "no list reached it");
  EXPECT(throws_with([&] { d.release(h_never); }, "not held"), "a second release must fail");
}

}  // namespace

int main(int argc, char **argv) {
  EXPECT(argc == 2, "usage: ds41_tiers_test <tensors.tsv>");
  namespace fs = std::filesystem;
  check_memlock();
  // A container's default memory lock is too small for the arena.
  setenv("ENGINE_UNLOCKED_ARENA", "1", 0);
  check_widest_ring(argv[1]);
  check_deadlines();
  std::printf("ring and deadlines pass\n");

  if (const std::string missing = missing_cpu_features(ENGINE_CPU_FEATURES); !missing.empty())
    return test::skip("the CPU lacks " + missing + ", which the tests' ggml build uses");
  ggml_cpu_init();

  std::string tmpl = (fs::current_path() / "ds41_tiers_test.XXXXXX").string();
  EXPECT(::mkdtemp(tmpl.data()) != nullptr, "cannot make a temporary directory");
  {
    const Generated g{fs::path(tmpl)};
    {
      DirectReader reader(g.paths);
      check_arena(g, reader);
      check_reserved(g, reader);
    }
    check_reads(g, staging_ring_bytes(g.largest(), STEP_MAX_ROWS));
    check_sums(g);
    check_read_failure(g);
    check_arena_failure(g);
    check_engine_thread(g);
    check_wait_bounds(g);
  }
  fs::remove_all(tmpl);
  std::printf("PASS: the arena, the reads, the sums, the pool jobs, the ring, the refusal and the deadline thread\n");
  return 0;
}
