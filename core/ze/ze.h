// The Level Zero layer: one driver, one context holding every card, memory, modules, kernels, command lists and
// events. Nothing above this file includes ze_api.h except through it. The features the design considered are
// mapped in docs/measurements/LEVEL_ZERO.md.
#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace engine::ze {

// Throws std::runtime_error naming the call and the result code.
void check(ze_result_t r, const char *what);
#define ZE_CHECK(call) ::engine::ze::check((call), #call)

// zeInit, with the cards' VMs out of page-fault mode (ADR-0037). The process's first zeInit fixes the mode, so every
// caller goes through this one.
ze_result_t init();

struct Card {
  ze_device_handle_t device = nullptr;
  std::string name;
  std::string pci;                 // bus address, "0000:03:00.0"
  std::string uuid;                // the device's UUID, in hex
  uint32_t driver_version = 0;     // the driver's driverVersion
  uint64_t memory = 0;             // device-local memory, bytes
  uint64_t max_alloc = 0;          // maxMemAllocSize
  uint32_t compute_ordinal = 0;    // command queue group with compute
  uint32_t copy_ordinal = UINT32_MAX;  // a copy-only group, if the card has one
  uint64_t timer_hz = 0;          // the global timestamp's ticks per second
  uint32_t clock_mhz = 0;         // coreClockRate: the top of the card's clock, which clock_read_device counts
  uint64_t timestamp_mask = 0;    // its valid bits
  uint32_t shared_system = 0;     // sharedSystemAllocCapabilities: how the card reaches the host's own memory
};

class Runtime {
 public:
  // Takes the first driver with GPUs, and every GPU it lists (ZE_AFFINITY_MASK narrows them). Throws when fewer
  // than min_cards are found.
  explicit Runtime(uint32_t min_cards = 1);
  ~Runtime();
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  ze_context_handle_t context() const { return ctx_; }
  const std::vector<Card> &cards() const { return cards_; }
  uint32_t ncards() const { return (uint32_t)cards_.size(); }

  // Device memory on one card; host memory in the context, which every card and the CPU read.
  void *alloc_device(uint32_t card, size_t bytes, size_t align = 256);
  void *alloc_host(size_t bytes, size_t align = 4096);
  void free(void *p);
  // Frees every device allocation on the card; host memory stays (a sleep: ADR-0032).
  void free_card(uint32_t card);
  uint64_t device_bytes(uint32_t card) const { return device_bytes_.at(card); }
  uint64_t host_bytes() const { return host_bytes_; }

  // Virtual memory (docs/measurements/LEVEL_ZERO.md, row 32): a reserved range of addresses whose blocks are mapped to device
  // memory and unmapped again, while the range stays where it is. vm_block is the granularity of both for large
  // ranges. A block of device memory backs one range at a time; vm_create counts it in device_bytes. free_card
  // unmaps and frees the card's ranges and blocks too.
  uint64_t vm_block(uint32_t card);
  void *vm_reserve(uint32_t card, size_t bytes);
  ze_physical_mem_handle_t vm_create(uint32_t card, size_t bytes);
  void vm_map(void *va, size_t bytes, ze_physical_mem_handle_t block);
  void vm_unmap(void *va, size_t bytes);

 private:
  ze_driver_handle_t driver_ = nullptr;
  ze_context_handle_t ctx_ = nullptr;
  std::vector<Card> cards_;
  std::vector<uint64_t> device_bytes_;
  uint64_t host_bytes_ = 0;
  std::map<void *, std::pair<int, size_t>> allocs_;  // card (-1: host), bytes
  std::map<void *, std::pair<int, size_t>> ranges_;  // reserved: card, bytes
  std::map<ze_physical_mem_handle_t, std::pair<int, size_t>> blocks_;  // card, bytes
  std::map<void *, size_t> mapped_;  // mapped address, bytes
};

class Kernel;

// A module built ahead of time by ocloc, loaded on one card.
class Module {
 public:
  Module(Runtime &rt, uint32_t card, const std::string &path);
  ~Module();
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  ze_kernel_handle_t kernel(const std::string &name);  // created once, owned by the module

 private:
  ze_module_handle_t mod_ = nullptr;
  std::map<std::string, ze_kernel_handle_t> kernels_;
};

// One kernel argument: a value of any size, or a pointer.
struct Arg {
  size_t size;
  const void *value;
  template <class T>
  Arg(const T &v) : size(sizeof(T)), value(&v) {}
};

// A pool of events every card and the host can signal and wait on.
class EventPool {
 public:
  EventPool(Runtime &rt, uint32_t count, bool timestamps = false);
  ~EventPool();
  EventPool(const EventPool &) = delete;
  EventPool &operator=(const EventPool &) = delete;
  ze_event_handle_t operator[](uint32_t i) const { return events_.at(i); }
  uint32_t size() const { return (uint32_t)events_.size(); }

 private:
  ze_event_pool_handle_t pool_ = nullptr;
  std::vector<ze_event_handle_t> events_;
};

// A command list on one card, either immediate (runs as appended) or regular (recorded, then executed as many times
// as wanted on the card's queue). Each command waits for the ones before it, but in a list made with in_order false
// the launches recorded between together(true) and together(false) may run at the same time, and the next command
// waits for all of them (docs/measurements/MEASUREMENTS.md#2026-10-16-decode-launches-together).
class List {
 public:
  enum Kind { immediate, regular };
  List(Runtime &rt, uint32_t card, Kind kind, bool copy_engine = false, bool in_order = true);
  ~List();
  List(const List &) = delete;
  List &operator=(const List &) = delete;

  // Sets the arguments in order and appends the launch. Group counts are in work-groups.
  void launch(ze_kernel_handle_t k, std::initializer_list<Arg> args, uint32_t gx, uint32_t gy, uint32_t gz,
              uint32_t lx, uint32_t ly = 1, uint32_t lz = 1, ze_event_handle_t signal = nullptr,
              std::initializer_list<ze_event_handle_t> wait = {});
  void copy(void *dst, const void *src, size_t bytes, ze_event_handle_t signal = nullptr,
            std::initializer_list<ze_event_handle_t> wait = {});
  // rows copies of width bytes in one command: row r from src + r * src_pitch to dst + r * dst_pitch.
  void copy_rows(void *dst, size_t dst_pitch, const void *src, size_t src_pitch, size_t width, size_t rows);
  void fill(void *dst, const void *pattern, size_t pattern_size, size_t bytes);
  void wait(std::initializer_list<ze_event_handle_t> events);
  void signal(ze_event_handle_t e);
  void reset(ze_event_handle_t e);
  void barrier();
  // Not in order only, and not while tracing: launches from together(true) to together(false) may overlap.
  void together(bool on);
  void timestamp(uint64_t *dst);  // the card's global timestamp, written when the list reaches it
  // Tracing (DESIGN.md, section 7): from here on, the card's timestamp after every launch into ts (host memory, cap
  // entries) and the kernel's name in trace_names(). ts[0] is written where trace() is called; launch i ends at
  // ts[i + 1].
  void trace(uint64_t *ts, size_t cap);
  void label(std::string name) { label_ = std::move(name); }  // names the next launch in the trace instead
  const std::vector<std::string> &trace_names() const { return trace_names_; }

  void close();         // regular lists: end recording
  void execute();       // regular lists: submit on the card's queue (does not wait)
  void synchronize();   // waits for everything submitted so far
  void reset_list();    // regular lists: empty it for recording again

  uint32_t card() const { return card_; }
  Kind kind() const { return kind_; }

 private:
  Runtime &rt_;
  uint32_t card_;
  Kind kind_;
  bool in_order_, together_ = false;
  ze_command_list_handle_t list_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  ze_fence_handle_t fence_ = nullptr;
  bool submitted_ = false;
  uint64_t *trace_ = nullptr;
  size_t trace_cap_ = 0;
  std::vector<std::string> trace_names_;
  std::string label_;
};

// Reads a whole file; throws if it cannot.
std::vector<uint8_t> read_file(const std::string &path);

}  // namespace engine::ze
