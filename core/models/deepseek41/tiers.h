// deepseek41's experts off the cards (ADR-0080 items 3-5): the anonymous arena, the O_DIRECT reader and its staging
// ring, the job that sums a layer's misses on the CPU, and the thread that bounds a list's wait on a host event.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cpu/experts.h"
#include "load/direct.h"
#include "models/deepseek41/layout.h"
#include "util/parallel.h"

namespace engine::deepseek41 {

constexpr uint64_t kArenaPieceBytes = 1ull << 30;
constexpr uint64_t kHugeBytes = 2ull << 20;
constexpr uint16_t kPartABlocks = 5, kPartBBlocks = 4;  // a down row's blocks in each card's part (ADR-0080 item 2)
constexpr std::chrono::seconds kHostWaitLimit{10};      // the longest the host sleeps on a read or a job

// Where a matrix of expert 0 starts: a shard's index and the byte in it.
struct MatrixPlace {
  uint32_t file = 0;
  uint64_t offset = 0;
};

// A layer's routed experts in the file. Gate and up share a type and a size, as every layer of the file does. Each
// matrix holds its experts one after another, so expert e starts e matrices' bytes in.
struct LayerExperts {
  ExpertShape shape;
  MatrixPlace gate, up, down;
  uint64_t gate_bytes = 0, down_bytes = 0;  // one expert's gate (and up), and its down
  uint32_t experts = D41_EXPERTS;
  uint64_t expert_bytes() const { return 2 * gate_bytes + down_bytes; }
};

// Anonymous memory under ADR-0035 and ADR-0037: 2 MiB aligned, MADV_HUGEPAGE, mlock2(MLOCK_ONFAULT). A memory lock too
// small for it throws, unless ENGINE_UNLOCKED_ARENA is set (docs/DEVELOPING.md); lock_errno() then says why it is not
// locked. Throws when mmap fails. reserved: the bytes come from the host's reserved huge pages (vm.nr_hugepages) when
// enough are free, which the kernel never moves or swaps; reserved() says whether they did.
class AnonMemory {
 public:
  explicit AnonMemory(uint64_t bytes, bool reserved = false);
  ~AnonMemory();
  AnonMemory(AnonMemory &&other) noexcept;
  AnonMemory(const AnonMemory &) = delete;
  AnonMemory &operator=(const AnonMemory &) = delete;
  AnonMemory &operator=(AnonMemory &&) = delete;

  uint8_t *data() const { return at_; }
  uint64_t bytes() const { return bytes_; }
  int lock_errno() const { return lock_errno_; }
  bool reserved() const { return reserved_; }

 private:
  void *map_ = nullptr;
  uint64_t map_bytes_ = 0, bytes_ = 0;
  uint8_t *at_ = nullptr;
  int lock_errno_ = 0;
  bool reserved_ = false;
};

// What ExpertArena::settle found.
struct ArenaReport {
  uint64_t pieces = 0, experts = 0, bytes = 0;
  uint64_t reserved_pieces = 0;          // pieces in the host's reserved huge pages
  uint64_t huge_ranges = 0, ranges = 0;  // the used 2 MiB ranges, and those in huge pages after MADV_COLLAPSE
  uint64_t touched_pages = 0;            // 4 KiB pages read once
  int lock_errno = 0;                    // why some piece is not locked; 0: every piece is
  double seconds = 0;
};

// The arena: pieces of anonymous memory, each expert whole inside one. An expert keeps gate and up whole in the file's
// layout, then down as part A's blocks of every row and then part B's. Addresses never move. Single-threaded, except
// that weights() may be read from any thread once filling is over.
class ExpertArena {
 public:
  // budget: the bytes experts may take, from the plan. piece_bytes: a multiple of kHugeBytes, and it maps virtual
  // memory only; a piece's unused part is never touched. reserved: pieces come from the host's reserved huge pages
  // while they last (Options::hugetlb_arena), which takes a piece whole.
  explicit ExpertArena(uint64_t budget, uint64_t piece_bytes = kArenaPieceBytes, bool reserved = false);

  // Reads experts ids of layer, in the order given, until the budget is met, on reader's pool and the calling thread.
  // Returns how many it placed: an expert that does not fit stays on the SSD, which is a normal plan. Throws when a read
  // fails, or the layer's down is not whole blocks of kPartABlocks + kPartBBlocks a row.
  uint32_t fill(DirectReader &reader, uint32_t layer, const LayerExperts &l, const std::vector<uint32_t> &ids);

  // Collapses each used range to a huge page and reads one byte of every used 4 KiB page. Run it after the file's pages
  // are given back and before a list binds the arena. Coverage comes from MADV_COLLAPSE and is best effort.
  ArenaReport settle();

  // Expert id of layer, with down_parts 2 and its part widths; gate is null when the arena does not hold it.
  ExpertWeights weights(uint32_t layer, uint32_t id) const;

  // An expert's change of tier, while no step runs. take forgets expert id of layer and returns its place, null when
  // the arena does not hold it. write puts an expert's matrices, in the file's layout, at such a place of the same
  // layer, from any thread while no expert is mapped there. put maps expert id of layer at the place.
  uint8_t *take(uint32_t layer, uint32_t id);
  static void write(const LayerExperts &l, const uint8_t *gate, const uint8_t *up, const uint8_t *down, uint8_t *to);
  void put(uint32_t layer, const LayerExperts &l, uint32_t id, uint8_t *place);
  const std::vector<AnonMemory> &pieces() const { return pieces_; }

 private:
  uint8_t *place(uint64_t need);
  static void write_down(const LayerExperts &l, const uint8_t *down, uint8_t *to);

  uint64_t budget_, piece_bytes_, used_ = 0, experts_ = 0, left_ = 0;
  bool reserved_;  // the next piece asks for reserved huge pages; cleared when a piece gets none
  std::vector<AnonMemory> pieces_;
  std::vector<uint64_t> piece_used_;
  std::vector<std::vector<ExpertWeights>> weights_;  // [layer][id]
};

// The bytes of the SSD reads' ring for steps of up to rows rows, as engine/models/deepseek41/memory.py's
// staging_ring_bytes has it: 2 layers x min(6 rows, 384) experts x (largest expert + 4 KiB a matrix).
uint64_t staging_ring_bytes(uint64_t largest_expert, uint32_t rows);

// A matrix's read in the ring: the ring offset of its first aligned block, and DirectReader::span of its bytes.
struct RingRead {
  uint64_t at = 0, span = 0;
};
struct RingExpert {
  RingRead gate, up, down;
};

// Where one round's reads land. A read costs up to two blocks more than its bytes, which on the file's offsets passes
// the ring's 4 KiB a matrix, so slots of a fixed size could overlap. Side 0 fills up from the ring's start and side 1
// down from its end, so two live rounds overlap only if their spans pass the whole ring, which place refuses.
class RingLayout {
 public:
  explicit RingLayout(uint64_t capacity);
  // Places expert id of l on side (0 or 1). Throws when it would reach the other side; nothing is overlapped.
  RingExpert place(uint32_t side, const LayerExperts &l, uint32_t id);
  void release(uint32_t side);  // the side's round is over: its bytes are free
  uint64_t capacity() const { return capacity_; }

 private:
  uint64_t capacity_, low_ = 0, high_;
};

// The SSD tier's reads: O_DIRECT in pieces of DirectReader::kPiece on its pool's threads, into an anonymous staging
// ring. One round at a time, started and waited for by one thread, never the engine thread. Refuses a file on a
// filesystem without O_DIRECT (ADR-0015 item 5).
class SsdReader {
 public:
  // paths: the shards, as MatrixPlace::file numbers them. ring_bytes: from staging_ring_bytes. wait_limit: wait's bound.
  SsdReader(const std::vector<std::string> &paths, uint64_t ring_bytes, unsigned threads = DirectReader::kThreads,
            std::chrono::nanoseconds wait_limit = kHostWaitLimit);
  // Waits for a running read however long it takes, since freeing the ring under it would corrupt memory.
  ~SsdReader();
  SsdReader(const SsdReader &) = delete;
  SsdReader &operator=(const SsdReader &) = delete;

  // Places the experts ids (distinct) on the side and starts their reads, every expert's three matrices at once, and
  // returns without waiting. out gets each expert's weights in the ring, whole down; read them only after wait. Throws
  // before any read when the ring cannot hold the round, or the last round is still running after wait gave up on it.
  void start(uint32_t side, const LayerExperts &l, const std::vector<uint32_t> &ids, std::vector<ExpertWeights> &out);
  // Sleeps until the round has landed; throws if a read failed, or if it has not landed within wait_limit. The round
  // then still runs, and wait may be called again. No round running returns at once.
  void wait();
  // The side's round is over: its bytes are free. Safe after wait gave up, since start refuses until the reads land.
  void release(uint32_t side) { layout_.release(side); }

  DirectReader &direct() { return reader_; }
  uint64_t rounds() const { return rounds_; }        // rounds started
  uint64_t pool_runs() const { return pool_runs_; }  // the pool's runs, after wait: one a round
  // Reads past 1 s, counted and logged as errors (ADR-0080 item 5's bound is near); the round still lands.
  uint64_t slow_reads() const { return slow_reads_; }
  // Whether a round's reads are in flight. Other readers of the file hold off while they are, since a miss waits for
  // the round and an expert takes the SSD about 1.7 ms alone and 2.5 beside two threads' refills
  // (docs/measurements/MEASUREMENTS.md#2026-11-13-deepseek41-refills-beside-miss-reads).
  bool busy() const { return busy_.load(std::memory_order_acquire); }
  // Called on the reading thread before each piece's read. For tests.
  void set_read_hook(std::function<void()> hook) { read_hook_ = std::move(hook); }

 private:
  struct Piece {
    uint32_t file;
    uint64_t offset, bytes;
    uint8_t *to;
  };
  void add_pieces(const MatrixPlace &first, uint64_t bytes, uint32_t id, const RingRead &read,
                  const uint8_t **weights);
  void round_thread();

  DirectReader reader_;
  AnonMemory ring_;
  RingLayout layout_;
  std::function<void()> read_hook_;
  std::vector<Piece> pieces_;
  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  const std::chrono::nanoseconds wait_limit_;
  bool stop_ = false, pending_ = false;
  std::string error_;
  uint64_t rounds_ = 0;
  std::atomic<uint64_t> pool_runs_{0}, slow_reads_{0}, pieces_left_{0};
  std::atomic<bool> busy_{false};
};

// One layer's misses for one step. jobs: the layer's records, one a row, read in place. rows: the step's recorded width
// and the rows of answer. real_rows: rows at or past it route to nothing, whatever their records hold. answer: rows
// x D41_EMBD f32, answer_stride floats apart. answer_flag, if not null, gets step behind a release fence once answer is
// written. A layer with no miss writes neither.
struct MissAsk {
  uint32_t layer = 0, rows = 0, real_rows = 0, step = 0;
  const D41MissJob *jobs = nullptr;
  float *answer = nullptr;
  uint64_t answer_stride = D41_EMBD;
  uint32_t *answer_flag = nullptr;
};

// What a layer's job did. A compute_many starts one CPU pool job, so a layer with both tiers wakes the workers twice
// (the SSD experts can start only once their reads land) and a layer with one tier once.
struct MissCounts {
  uint32_t routings = 0;      // the misses of the layer's real rows
  uint32_t arena_jobs = 0;    // those served from the arena
  uint32_t ssd_jobs = 0;      // those served from the ring
  uint32_t ssd_experts = 0;   // distinct experts read: an expert several rows route to is read once
  uint32_t read_rounds = 0;   // 0 or 1
  uint32_t cpu_pool_runs = 0;
};

// A layer's misses: the arena experts' sum and then the SSD experts', each in routing order (row by row, then rank by
// rank), so the bits are the same on every run. The reads overlap the arena's sum. run_layer works on the calling
// thread; submit hands the job to this object's own thread, which sleeps between jobs, so the engine thread never waits
// on a read. Use one or the other.
class MissJobs {
 public:
  // layers: by layer number. arena and reader must outlive it. cpu: its workers run the sums, with the calling thread.
  // wait_limit: wait's bound.
  MissJobs(std::vector<LayerExperts> layers, const ExpertArena &arena, SsdReader &reader, Pool &cpu,
           std::chrono::nanoseconds wait_limit = kHostWaitLimit);
  ~MissJobs();
  MissJobs(const MissJobs &) = delete;
  MissJobs &operator=(const MissJobs &) = delete;

  // Throws on a bad record, a read that failed or passed the reader's wait_limit, or a ring that cannot hold the round;
  // the flag is then not raised, the card's bounded wait gives up and the step fails (ADR-0080 item 5).
  MissCounts run_layer(const MissAsk &ask);
  void submit(const MissAsk &ask);  // takes no lock longer than a queue push, never reads
  // Sleeps until every submitted job is done. Throws the first failure since the last wait, naming its layer. Past
  // wait_limit it drops the jobs not begun and throws, naming the layer; the job running goes on, so what its ask points
  // to must stay valid until a later wait returns.
  void wait();
  // Called on the job's thread just before the arena sum starts, and when it ends, reads possibly still running. For
  // tests.
  void set_arena_start_hook(std::function<void()> hook) { arena_start_hook_ = std::move(hook); }
  void set_arena_hook(std::function<void()> hook) { arena_hook_ = std::move(hook); }

 private:
  struct SsdJob {
    uint32_t expert, tok;  // expert: index into ssd_ids_
    float w;
  };
  void job_thread();

  std::vector<LayerExperts> layers_;
  const ExpertArena &arena_;
  SsdReader &reader_;
  Pool &pool_;
  CpuExperts cpu_;
  std::function<void()> arena_start_hook_, arena_hook_;

  std::vector<CpuExperts::Job> arena_jobs_, ssd_jobs_;
  std::vector<SsdJob> ssd_refs_;
  std::vector<uint32_t> ssd_ids_;
  std::vector<ExpertWeights> ssd_weights_;
  std::vector<float> arena_y_, ssd_y_;

  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_, idle_;
  std::deque<MissAsk> queue_;
  const std::chrono::nanoseconds wait_limit_;
  bool stop_ = false, busy_ = false;
  bool abandoned_ = false;  // wait gave up on the running job, so its failure has been reported
  uint32_t running_layer_ = 0;
  std::string error_;
};

// A host-signalled event a list waits on, as the deadline thread sees it: the small interface a card's event and a
// test's fake share.
class DeadlineEvent {
 public:
  virtual ~DeadlineEvent() = default;
  virtual bool reached() = 0;  // the list has reached its wait on this event
  virtual void signal() = 0;   // lets the list go on; may be called after the host signalled it too
};

// The host deadline thread of ADR-0080 item 5. It holds events; limit after a held event's list reaches it, it sets
// the event's stuck flag and then signals it, so the list goes on and the step fails cleanly. Sleeps while it holds
// nothing and polls a millisecond apart otherwise.
class HostDeadlines {
 public:
  explicit HostDeadlines(std::chrono::nanoseconds limit = std::chrono::seconds(1));
  ~HostDeadlines();
  HostDeadlines(const HostDeadlines &) = delete;
  HostDeadlines &operator=(const HostDeadlines &) = delete;

  // Holds ev until release; ev and stuck must live until then. Returns the hold's id.
  uint64_t hold(DeadlineEvent &ev, uint32_t *stuck);
  // Ends a hold, before or after the host signals the event; afterwards the thread touches neither. Returns whether
  // the deadline passed, in which case stuck is set and the event signalled.
  bool release(uint64_t id);

 private:
  struct Hold {
    uint64_t id;
    DeadlineEvent *ev;
    uint32_t *stuck;
    bool seen, fired;
    std::chrono::steady_clock::time_point at;  // when the list was first seen at the event
  };
  void run();

  const std::chrono::nanoseconds limit_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Hold> holds_;
  uint64_t next_ = 1;
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace engine::deepseek41
