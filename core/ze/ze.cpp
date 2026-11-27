#include "ze.h"

#include "library.h"

#include <sys/mman.h>
#include <sys/stat.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace engine::ze {

// Set when a list times out or the device is lost. The driver's destroy and free calls then wait for a queue that
// never finishes, so the destructors below make none: the process is to end, and the kernel takes back what the cards
// hold.
static std::atomic<bool> stuck{false};

static std::string hex_result(ze_result_t r) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "0x%x", (unsigned)r);
  return buf;
}

// A list on a queue whose engine the driver reset never finishes, and the driver's job timeout does not apply to the
// engine's long-running queues, so an unbounded wait hangs the process. No list the engine records runs this long.
constexpr uint64_t kListWaitNs = 60'000'000'000ull;

bool queue_stuck() { return stuck; }
void mark_stuck() { stuck = true; }

void check(ze_result_t r, const char *what) {
  if (r == ZE_RESULT_SUCCESS) return;
  if (r == ZE_RESULT_ERROR_DEVICE_LOST) stuck = true;
  throw std::runtime_error(std::string(what) + " returned " + hex_result(r));
}

bool check_peer_probe(ze_result_t r, const std::string &what, std::string *why, bool timed) {
  if (r == ZE_RESULT_SUCCESS) return true;
  if (r == ZE_RESULT_ERROR_DEVICE_LOST) check(r, what.c_str());
  if (timed && r == ZE_RESULT_NOT_READY) {
    stuck = true;
    throw std::runtime_error(what + " did not finish in " + std::to_string(kListWaitNs / 1'000'000'000ull) + " s");
  }
  if (why) *why = what + " returned " + hex_result(r);
  return false;
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static bool env_override(const char *name, bool current) {
  const char *v = std::getenv(name);
  if (!v || !*v) return current;
  return std::strcmp(v, "0") != 0;
}

#if defined(__x86_64__)
__attribute__((target("clflushopt"))) static void clflushopt_range(const char *p, size_t bytes) {
  constexpr size_t kLine = 64;
  uintptr_t cur = reinterpret_cast<uintptr_t>(p) & ~(kLine - 1);
  const uintptr_t end = reinterpret_cast<uintptr_t>(p) + bytes;
  for (; cur < end; cur += kLine) _mm_clflushopt(reinterpret_cast<void *>(cur));
}

static void clflush_range(const char *p, size_t bytes) {
  constexpr size_t kLine = 64;
  uintptr_t cur = reinterpret_cast<uintptr_t>(p) & ~(kLine - 1);
  const uintptr_t end = reinterpret_cast<uintptr_t>(p) + bytes;
  for (; cur < end; cur += kLine) _mm_clflush(reinterpret_cast<const void *>(cur));
}
#endif

void flush_cpu_writes(const void *p, size_t bytes) {
#if defined(__x86_64__)
  if (!p || bytes == 0) return;
  static const bool has_clflushopt = __builtin_cpu_supports("clflushopt");
  if (has_clflushopt)
    clflushopt_range(static_cast<const char *>(p), bytes);
  else
    clflush_range(static_cast<const char *>(p), bytes);
  _mm_sfence();
#else
  (void)p;
  (void)bytes;
#endif
}

uint8_t readback_mapped_byte(const void *p, bool writeback) {
#if defined(__x86_64__)
  if (writeback) flush_cpu_writes(p, 1);
  _mm_mfence();
#else
  (void)writeback;
#endif
  return *static_cast<volatile const uint8_t *>(p);
}

ze_result_t init() {
  // compute-runtime reads these keys only with NEOReadDebugKeys set. A value already in the environment wins.
  setenv("NEOReadDebugKeys", "1", 0);
  setenv("NEO_EnableRecoverablePageFaults", "0", 0);
  return zeInit(0);
}

Runtime::Runtime(uint32_t min_cards) {
  ZE_CHECK(init());
  uint32_t nd = 0;
  ZE_CHECK(zeDriverGet(&nd, nullptr));
  std::vector<ze_driver_handle_t> drivers(nd);
  ZE_CHECK(zeDriverGet(&nd, drivers.data()));
  for (auto drv : drivers) {
    uint32_t n = 0;
    ZE_CHECK(zeDeviceGet(drv, &n, nullptr));
    std::vector<ze_device_handle_t> devs(n);
    ZE_CHECK(zeDeviceGet(drv, &n, devs.data()));
    std::vector<Card> found;
    for (auto dev : devs) {
      ze_device_ip_version_ext_t ipv = {ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT};
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, &ipv};
      ZE_CHECK(zeDeviceGetProperties(dev, &p));
      if (p.type != ZE_DEVICE_TYPE_GPU) continue;
      Card c;
      c.device = dev;
      c.name = p.name;
      c.max_alloc = p.maxMemAllocSize;
      c.device_id = p.deviceId;
      c.ip_version = ipv.ipVersion;
      c.host_flag_polling = !env_override("ENGINE_NO_HOST_POLL", c.is_hpg());
      c.writeback_mapping = env_override("ENGINE_STAGE_FLUSH", c.is_hpg());
      c.copy_2d_arena = !env_override("ENGINE_1D_STAGE_COPY", c.is_hpg());
      c.xe_cores = p.numSlices * p.numSubslicesPerSlice;
      c.timer_hz = p.timerResolution;  // cycles per second, with the 1.2 properties
      c.timestamp_mask = p.timestampValidBits >= 64 ? ~0ull : ((1ull << p.timestampValidBits) - 1);
      uint32_t nm = 0;
      ZE_CHECK(zeDeviceGetMemoryProperties(dev, &nm, nullptr));
      std::vector<ze_device_memory_properties_t> mp(nm, {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES});
      std::vector<ze_device_memory_ext_properties_t> mx(nm, {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES});
      for (uint32_t i = 0; i < nm; i++) mp[i].pNext = &mx[i];
      ZE_CHECK(zeDeviceGetMemoryProperties(dev, &nm, mp.data()));
      // totalSize is what the driver offers one process, a share of the memory. A discrete card's physical size is what
      // it holds before it spills into host memory. Zero says the driver does not know it.
      const bool integrated = p.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;
      for (uint32_t i = 0; i < nm; i++) c.memory += !integrated && mx[i].physicalSize ? mx[i].physicalSize : mp[i].totalSize;
      ze_device_memory_access_properties_t ma = {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
      ZE_CHECK(zeDeviceGetMemoryAccessProperties(dev, &ma));
      c.shared_system = ma.sharedSystemAllocCapabilities;
      ze_pci_ext_properties_t pci = {ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
      if (zeDevicePciGetPropertiesExt(dev, &pci) == ZE_RESULT_SUCCESS) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x", pci.address.domain, pci.address.bus, pci.address.device,
                      pci.address.function);
        c.pci = buf;
        c.pcie_gen = pci.maxSpeed.genVersion;
        c.pcie_width = pci.maxSpeed.width;
        c.pcie_bytes_per_s = pci.maxSpeed.maxBandwidth;
      }
      uint32_t ng = 0;
      ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &ng, nullptr));
      std::vector<ze_command_queue_group_properties_t> gp(ng, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
      ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &ng, gp.data()));
      bool have_compute = false;
      for (uint32_t g = 0; g < ng; g++) {
        bool compute = gp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        bool copy = gp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        if (compute && !have_compute) c.compute_ordinal = g, have_compute = true;
        if (copy && !compute && c.copy_ordinal == UINT32_MAX) c.copy_ordinal = g;
      }
      if (!have_compute) throw std::runtime_error(c.name + ": no compute queue group");
      found.push_back(c);
    }
    if (!found.empty()) {
      driver_ = drv;
      cards_ = std::move(found);
      break;
    }
  }
  if (cards_.size() < min_cards)
    throw std::runtime_error("found " + std::to_string(cards_.size()) + " GPU(s), need " + std::to_string(min_cards));
  std::vector<ze_device_handle_t> devs;
  for (auto &c : cards_) devs.push_back(c.device);
  ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ZE_CHECK(zeContextCreateEx(driver_, &cd, (uint32_t)devs.size(), devs.data(), &ctx_));
  device_bytes_.assign(cards_.size(), 0);
}

Runtime::~Runtime() {
  if (stuck) return;
  for (auto &[p, m] : cpu_maps_) munmap(m.first, m.second);
  for (auto &[va, bytes] : mapped_) zeVirtualMemUnmap(ctx_, va, bytes);
  for (auto &[b, info] : blocks_) zePhysicalMemDestroy(ctx_, b);
  for (auto &[va, info] : ranges_) zeVirtualMemFree(ctx_, va, info.second);
  for (auto &[p, info] : allocs_) zeMemFree(ctx_, p);
  if (ctx_) zeContextDestroy(ctx_);
}

uint64_t Runtime::vm_block(uint32_t card) {
  size_t page = 0;
  ZE_CHECK(zeVirtualMemQueryPageSize(ctx_, cards_.at(card).device, (size_t)1 << 30, &page));
  return page;
}

void *Runtime::vm_reserve(uint32_t card, size_t bytes) {
  void *va = nullptr;
  ZE_CHECK(zeVirtualMemReserve(ctx_, nullptr, bytes, &va));
  ranges_[va] = {(int)card, bytes};
  return va;
}

ze_physical_mem_handle_t Runtime::vm_create(uint32_t card, size_t bytes) {
  ze_physical_mem_desc_t d = {ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC};
  d.size = bytes;
  ze_physical_mem_handle_t b = nullptr;
  ze_result_t r = zePhysicalMemCreate(ctx_, cards_.at(card).device, &d, &b);
  if (r != ZE_RESULT_SUCCESS)
    throw std::runtime_error("zePhysicalMemCreate of " + std::to_string(bytes) + " B on card " + std::to_string(card) +
                             " failed (" + std::to_string((unsigned)r) + "); " +
                             std::to_string(device_bytes_[card]) + " B allocated before it");
  blocks_[b] = {(int)card, bytes};
  device_bytes_[card] += bytes;
  return b;
}

void Runtime::vm_map(void *va, size_t bytes, ze_physical_mem_handle_t block) {
  ZE_CHECK(zeVirtualMemMap(ctx_, va, bytes, block, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
  mapped_[va] = bytes;
}

void Runtime::vm_unmap(void *va, size_t bytes) {
  ZE_CHECK(zeVirtualMemUnmap(ctx_, va, bytes));
  mapped_.erase(va);
}

void *Runtime::alloc_device(uint32_t card, size_t bytes, size_t align) {
  ze_device_mem_alloc_desc_t d = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ze_relaxed_allocation_limits_exp_desc_t relaxed = {ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC};
  relaxed.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
  if (bytes > cards_.at(card).max_alloc) d.pNext = &relaxed;
  void *p = nullptr;
  ze_result_t r = zeMemAllocDevice(ctx_, &d, bytes ? bytes : 1, align, cards_.at(card).device, &p);
  if (r != ZE_RESULT_SUCCESS)
    throw std::runtime_error("zeMemAllocDevice of " + std::to_string(bytes) + " B on card " + std::to_string(card) +
                             " failed (" + std::to_string((unsigned)r) + "); " +
                             std::to_string(device_bytes_[card]) + " B allocated before it");
  allocs_[p] = {(int)card, bytes};
  device_bytes_[card] += bytes;
  return p;
}

void *Runtime::alloc_device_mapped(uint32_t card, size_t bytes, uint8_t *&cpu, size_t align) {
  auto fail = [&](const std::string &what) {
    return std::runtime_error("alloc_device_mapped of " + std::to_string(bytes) + " B on card " + std::to_string(card) +
                              ": " + what + " (ADR-0065)");
  };
  ze_external_memory_export_desc_t ex = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC, nullptr,
                                         ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF};
  ze_relaxed_allocation_limits_exp_desc_t relaxed = {ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC};
  relaxed.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
  if (bytes > cards_.at(card).max_alloc) ex.pNext = &relaxed;
  ze_device_mem_alloc_desc_t d = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  d.pNext = &ex;
  void *p = nullptr;
  if (const ze_result_t r = zeMemAllocDevice(ctx_, &d, bytes ? bytes : 1, align, cards_.at(card).device, &p);
      r != ZE_RESULT_SUCCESS)
    throw fail("zeMemAllocDevice with a dma-buf export failed (" + std::to_string((unsigned)r) + "); " +
               std::to_string(device_bytes_[card]) + " B allocated before it");
  allocs_[p] = {(int)card, bytes};
  device_bytes_[card] += bytes;
  ze_external_memory_export_fd_t fd = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD, nullptr,
                                       ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, -1};
  ze_memory_allocation_properties_t props = {ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, &fd};
  // The runtime owns this descriptor: it returns the same number each time it is asked and closes it in zeMemFree
  // (docs/measurements/MEASUREMENTS.md#2026-11-05-staging-through-a-mapping). So the engine never closes it, on the
  // failure paths too: closing it would make zeMemFree close whatever file has that number by then.
  if (zeMemGetAllocProperties(ctx_, p, &props, nullptr) != ZE_RESULT_SUCCESS || fd.fd < 0) {
    free(p);
    throw fail("the runtime gave no dma-buf descriptor");
  }
  void *m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd.fd, 0);
  if (m == MAP_FAILED) {
    const int e = errno;
    free(p);
    throw fail(std::string("mmap of its dma-buf failed: ") + std::strerror(e));
  }
  cpu = static_cast<uint8_t *>(m);
  cpu_maps_[p] = {m, bytes};
  return p;
}

void Runtime::unmap_cpu(void *p) {
  auto m = cpu_maps_.find(p);
  if (m == cpu_maps_.end()) return;
  munmap(m->second.first, m->second.second);
  cpu_maps_.erase(m);
}

void *Runtime::alloc_host(size_t bytes, size_t align) {
  ze_host_mem_alloc_desc_t d = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void *p = nullptr;
  ze_result_t r = zeMemAllocHost(ctx_, &d, bytes ? bytes : 1, align, &p);
  if (r != ZE_RESULT_SUCCESS)
    throw std::runtime_error("zeMemAllocHost of " + std::to_string(bytes) + " B failed (" +
                             std::to_string((unsigned)r) + "); " + std::to_string(host_bytes_) +
                             " B allocated before it");
  allocs_[p] = {-1, bytes};
  host_bytes_ += bytes;
  return p;
}

void Runtime::free(void *p) {
  auto it = allocs_.find(p);
  if (it == allocs_.end()) throw std::runtime_error("free of memory the runtime did not allocate");
  unmap_cpu(p);
  if (it->second.first < 0)
    host_bytes_ -= it->second.second;
  else
    device_bytes_[it->second.first] -= it->second.second;
  allocs_.erase(it);
  ZE_CHECK(zeMemFree(ctx_, p));
}

void Runtime::free_card(uint32_t card) {
  for (auto r = ranges_.begin(); r != ranges_.end();) {
    if (r->second.first != (int)card) {
      ++r;
      continue;
    }
    const auto hi = (uint8_t *)r->first + r->second.second;
    for (auto m = mapped_.lower_bound(r->first); m != mapped_.end() && (uint8_t *)m->first < hi;) {
      ZE_CHECK(zeVirtualMemUnmap(ctx_, m->first, m->second));
      m = mapped_.erase(m);
    }
    ZE_CHECK(zeVirtualMemFree(ctx_, r->first, r->second.second));
    r = ranges_.erase(r);
  }
  for (auto b = blocks_.begin(); b != blocks_.end();) {
    if (b->second.first != (int)card) {
      ++b;
      continue;
    }
    ZE_CHECK(zePhysicalMemDestroy(ctx_, b->first));
    device_bytes_[card] -= b->second.second;
    b = blocks_.erase(b);
  }
  for (auto it = allocs_.begin(); it != allocs_.end();) {
    if (it->second.first != (int)card) {
      ++it;
      continue;
    }
    unmap_cpu(it->first);
    ZE_CHECK(zeMemFree(ctx_, it->first));
    device_bytes_[card] -= it->second.second;
    it = allocs_.erase(it);
  }
}

std::string Card::ip() const {
  if (!ip_version) return "";
  return std::to_string(ip_version >> 22) + "." + std::to_string((ip_version >> 14) & 0xff) + "." +
         std::to_string(ip_version & 0x3f);
}

Module::Module(Runtime &rt, uint32_t card, const std::string &path, const std::string *spirv_options) {
  auto bin = read_file(path);
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC};
  md.format = spirv_options ? ZE_MODULE_FORMAT_IL_SPIRV : ZE_MODULE_FORMAT_NATIVE;
  md.inputSize = bin.size();
  md.pInputModule = bin.data();
  if (spirv_options) md.pBuildFlags = spirv_options->c_str();
  ze_module_build_log_handle_t log = nullptr;
  ze_result_t r = zeModuleCreate(rt.context(), rt.cards().at(card).device, &md, &mod_, &log);
  if (r != ZE_RESULT_SUCCESS) {
    size_t n = 0;
    zeModuleBuildLogGetString(log, &n, nullptr);
    std::string s(n, '\0');
    zeModuleBuildLogGetString(log, &n, s.data());
    zeModuleBuildLogDestroy(log);
    throw std::runtime_error("zeModuleCreate(" + path + ") failed: " + s);
  }
  if (log) zeModuleBuildLogDestroy(log);
}

std::string select_spirv_dir(const Card &card, const std::string &xe2_dir, const std::string &hpg_dir) {
  const bool hpg = card.is_hpg();
  const std::string &dir = hpg ? hpg_dir : xe2_dir;
  if (dir.empty()) {
    const std::string ip = card.ip();
    throw std::runtime_error("card (" + card.name + ", IP " + (ip.empty() ? "not reported" : ip) +
                             ") needs " + (hpg ? "Xe-HPG (Alchemist)" : "Xe2 (Battlemage)") +
                             " SPIR-V, which this build omitted (add " +
                             (hpg ? "acm-g10/acm-g11/acm-g12" : "bmg-g21/bmg-g31") + " to ENGINE_AOT_DEVICES)");
  }
  return dir;
}

Library::Library(Runtime &rt, std::string dir, std::string spirv_dir, std::string hpg_spirv_dir)
    : rt_(rt),
      dir_(std::move(dir)),
      spirv_dir_(std::move(spirv_dir)),
      hpg_spirv_dir_(std::move(hpg_spirv_dir)) {
  const char *target = std::getenv("ENGINE_KERNEL_TARGET");
  const bool force = target && std::string(target) == "spirv";
  for (uint32_t c = 0; c < rt_.ncards(); c++) {
    const Card &card = rt_.cards()[c];
    struct stat st;
    const std::string ip = card.ip();
    const bool built = !ip.empty() && ::stat((dir_ + "/" + ip).c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    const bool need_spirv = force || !built;
    spirv_.push_back(need_spirv);
    if (force)
      std::fprintf(stderr, "engine: card %u: ENGINE_KERNEL_TARGET=spirv, its kernels are compiled at load\n", c);
    else if (!built)
      std::fprintf(stderr,
                   "engine: card %u (%s, IP %s): no kernels are built for this chip, which is untested; its kernels "
                   "are compiled from SPIR-V at load\n",
                   c, card.name.c_str(), ip.empty() ? "not reported" : ip.c_str());
    if (need_spirv) (void)select_spirv_dir(card, spirv_dir_, hpg_spirv_dir_);
  }
}

ze_kernel_handle_t Library::get(uint32_t card, const std::string &module, const std::string &name) {
  auto &m = modules_[{card, module}];
  if (!m) {
    const Card &c = rt_.cards().at(card);
    if (spirv_.at(card)) {
      auto opts = read_file(dir_ + "/" + module + ".options");
      const std::string options(opts.begin(), opts.end());
      const std::string spv_dir = select_spirv_dir(c, spirv_dir_, hpg_spirv_dir_);
      m = std::make_unique<Module>(rt_, card, spv_dir + "/" + module + ".spv", &options);
    } else {
      m = std::make_unique<Module>(rt_, card, dir_ + "/" + c.ip() + "/" + module + ".bin");
    }
  }
  return m->kernel(name);
}

Module::~Module() {
  if (stuck) return;
  for (auto &[n, k] : kernels_) zeKernelDestroy(k);
  if (mod_) zeModuleDestroy(mod_);
}

ze_kernel_handle_t Module::kernel(const std::string &name) {
  auto it = kernels_.find(name);
  if (it != kernels_.end()) return it->second;
  ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
  kd.pKernelName = name.c_str();
  ze_kernel_handle_t k = nullptr;
  ze_result_t r = zeKernelCreate(mod_, &kd, &k);
  if (r != ZE_RESULT_SUCCESS) throw std::runtime_error("zeKernelCreate(" + name + ") failed");
  kernels_[name] = k;
  return k;
}

EventPool::EventPool(Runtime &rt, uint32_t count, bool timestamps) {
  ze_event_pool_desc_t pd = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | (timestamps ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0);
  pd.count = count;
  std::vector<ze_device_handle_t> devs;
  for (auto &c : rt.cards()) devs.push_back(c.device);
  ZE_CHECK(zeEventPoolCreate(rt.context(), &pd, (uint32_t)devs.size(), devs.data(), &pool_));
  events_.resize(count);
  for (uint32_t i = 0; i < count; i++) {
    ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC};
    ed.index = i;
    ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    ZE_CHECK(zeEventCreate(pool_, &ed, &events_[i]));
  }
}

EventPool::~EventPool() {
  if (stuck) return;
  for (auto e : events_) zeEventDestroy(e);
  if (pool_) zeEventPoolDestroy(pool_);
}

Queue::Queue(Runtime &rt, uint32_t card, bool copy_engine, bool in_order)
    : card_(card), in_order_(in_order) {
  const Card &c = rt.cards().at(card);
  ordinal_ = copy_engine && c.copy_ordinal != UINT32_MAX ? c.copy_ordinal : c.compute_ordinal;
  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = ordinal_;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qd.flags = 0;
  ZE_CHECK(zeCommandQueueCreate(rt.context(), c.device, &qd, &queue_));
}

Queue::~Queue() {
  if (!stuck && queue_) zeCommandQueueDestroy(queue_);
}

List::List(Runtime &rt, uint32_t card, Kind kind, bool copy_engine, bool in_order)
    : rt_(rt), card_(card), kind_(kind), in_order_(in_order) {
  const Card &c = rt.cards().at(card);
  ordinal_ = copy_engine && c.copy_ordinal != UINT32_MAX ? c.copy_ordinal : c.compute_ordinal;
  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = ordinal_;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qd.flags = kind == immediate && in_order ? ZE_COMMAND_QUEUE_FLAG_IN_ORDER : 0;
  if (kind == immediate) {
    ZE_CHECK(zeCommandListCreateImmediate(rt.context(), c.device, &qd, &list_));
  } else {
    ZE_CHECK(zeCommandQueueCreate(rt.context(), c.device, &qd, &queue_));
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ld.commandQueueGroupOrdinal = ordinal_;
    ld.flags = in_order ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0;
    ZE_CHECK(zeCommandListCreate(rt.context(), c.device, &ld, &list_));
    ze_fence_desc_t fd = {ZE_STRUCTURE_TYPE_FENCE_DESC};
    ZE_CHECK(zeFenceCreate(queue_, &fd, &fence_));
    parts_.push_back({list_, fence_, false});
  }
}

List::List(Runtime &rt, Queue &queue, bool trace)
    : rt_(rt),
      card_(queue.card()),
      ordinal_(queue.ordinal()),
      kind_(regular),
      in_order_(queue.in_order()),
      owns_queue_(false),
      queue_(queue.handle()) {
  (void)trace;
  const Card &c = rt.cards().at(card_);
  ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  ld.commandQueueGroupOrdinal = ordinal_;
  ld.flags = in_order_ ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0;
  ZE_CHECK(zeCommandListCreate(rt.context(), c.device, &ld, &list_));
  ze_fence_desc_t fd = {ZE_STRUCTURE_TYPE_FENCE_DESC};
  ZE_CHECK(zeFenceCreate(queue_, &fd, &fence_));
  parts_.push_back({list_, fence_, false});
}

#if defined(ENGINE_CHECK_LISTS) || !defined(NDEBUG)
#define ONE_USER OneUser one_user_(*this)
#else
#define ONE_USER (void)0
#endif

List::OneUser::OneUser(List &list) : l(list) {
  const std::thread::id me = std::this_thread::get_id();
  std::thread::id none{};
  if (l.user_.compare_exchange_strong(none, me)) {
    outer = true;
  } else if (none != me) {
    throw std::logic_error("card " + std::to_string(l.card_) + ": a command list called from two threads at once");
  }
}
List::OneUser::~OneUser() {
  if (outer) l.user_.store(std::thread::id{});
}

List::~List() {
  if (stuck) return;
  if (kind_ == immediate) {
    if (list_) zeCommandListDestroy(list_);
  } else {
    for (Part &p : parts_) {
      if (p.fence) zeFenceDestroy(p.fence);
      if (p.list) zeCommandListDestroy(p.list);
    }
    for (ze_event_handle_t e : split_events_) zeEventDestroy(e);
    for (ze_event_pool_handle_t pool : split_pools_) zeEventPoolDestroy(pool);
  }
  if (owns_queue_ && queue_) zeCommandQueueDestroy(queue_);
}

void List::launch(ze_kernel_handle_t k, std::initializer_list<Arg> args, uint32_t gx, uint32_t gy, uint32_t gz,
                  uint32_t lx, uint32_t ly, uint32_t lz, ze_event_handle_t signal,
                  std::initializer_list<ze_event_handle_t> wait) {
  ONE_USER;
  uint32_t i = 0;
  for (const Arg &a : args) {
    ze_result_t r = zeKernelSetArgumentValue(k, i, a.size, a.value);
    if (r != ZE_RESULT_SUCCESS) {
      size_t len = 0;
      zeKernelGetName(k, &len, nullptr);
      std::string name(len, '\0');
      zeKernelGetName(k, &len, name.data());
      check(r, ("zeKernelSetArgumentValue(" + std::string(name.c_str()) + ", argument " + std::to_string(i) + ")").c_str());
    }
    i++;
  }
  ZE_CHECK(zeKernelSetGroupSize(k, lx, ly, lz));
  ze_group_count_t g = {gx, gy, gz};
  std::vector<ze_event_handle_t> w(wait);
  ZE_CHECK(zeCommandListAppendLaunchKernel(list_, k, &g, signal, (uint32_t)w.size(), w.empty() ? nullptr : w.data()));
  if (!in_order_ && (!together_ || trace_)) barrier();
  if (trace_ && trace_names_.size() + 1 < trace_cap_) {
    size_t len = 0;
    ZE_CHECK(zeKernelGetName(k, &len, nullptr));
    std::string name(len, '\0');
    ZE_CHECK(zeKernelGetName(k, &len, name.data()));
    if (!name.empty() && name.back() == '\0') name.pop_back();
    trace_names_.push_back(label_.empty() ? name : label_);
    timestamp(trace_ + trace_names_.size());
  }
  label_.clear();
}

void List::trace(uint64_t *ts, size_t cap) {
  ONE_USER;
  trace_ = ts;
  trace_cap_ = cap;
  trace_names_.clear();
  timestamp(ts);
}

void List::copy(void *dst, const void *src, size_t bytes, ze_event_handle_t signal,
                std::initializer_list<ze_event_handle_t> wait) {
  ONE_USER;
  std::vector<ze_event_handle_t> w(wait);
  ZE_CHECK(zeCommandListAppendMemoryCopy(list_, dst, src, bytes, signal, (uint32_t)w.size(),
                                         w.empty() ? nullptr : w.data()));
  if (!in_order_) barrier();
}

void List::copy_rows(void *dst, size_t dst_pitch, const void *src, size_t src_pitch, size_t width, size_t rows) {
  ONE_USER;
  const ze_copy_region_t region = {0, 0, 0, (uint32_t)width, (uint32_t)rows, 1};
  ZE_CHECK(zeCommandListAppendMemoryCopyRegion(list_, dst, &region, (uint32_t)dst_pitch, 0, src, &region,
                                               (uint32_t)src_pitch, 0, nullptr, 0, nullptr));
  if (!in_order_) barrier();
}

void List::fill(void *dst, const void *pattern, size_t pattern_size, size_t bytes) {
  ONE_USER;
  ZE_CHECK(zeCommandListAppendMemoryFill(list_, dst, pattern, pattern_size, bytes, nullptr, 0, nullptr));
  if (!in_order_) barrier();
}

void List::wait(std::initializer_list<ze_event_handle_t> events) {
  ONE_USER;
  std::vector<ze_event_handle_t> w(events);
  ZE_CHECK(zeCommandListAppendWaitOnEvents(list_, (uint32_t)w.size(), w.data()));
}

void List::signal(ze_event_handle_t e) {
  ONE_USER;
  ZE_CHECK(zeCommandListAppendSignalEvent(list_, e));
}
void List::reset(ze_event_handle_t e) {
  ONE_USER;
  ZE_CHECK(zeCommandListAppendEventReset(list_, e));
}
void List::barrier() {
  ONE_USER;
  ZE_CHECK(zeCommandListAppendBarrier(list_, nullptr, 0, nullptr));
}

void List::together(bool on) {
  ONE_USER;
  if (in_order_) return;
  if (!on && together_ && !trace_) barrier();  // while tracing, each launch already had its own
  together_ = on;
}

void List::timestamp(uint64_t *dst) {
  ONE_USER;
  ZE_CHECK(zeCommandListAppendWriteGlobalTimestamp(list_, dst, nullptr, 0, nullptr));
}

void List::split() {
  ONE_USER;
  if (kind_ != regular) throw std::runtime_error("split() on an immediate list");
  constexpr uint32_t kSplitPoolEvents = 64;
  const uint32_t idx = static_cast<uint32_t>(split_events_.size() % kSplitPoolEvents);
  if (idx == 0) {
    ze_event_pool_desc_t pd = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pd.count = kSplitPoolEvents;
    ze_event_pool_handle_t pool = nullptr;
    ze_device_handle_t dev = rt_.cards().at(card_).device;
    ZE_CHECK(zeEventPoolCreate(rt_.context(), &pd, 1, &dev, &pool));
    split_pools_.push_back(pool);
  }
  ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC};
  ed.index = idx;
  ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_event_handle_t ev = nullptr;
  ZE_CHECK(zeEventCreate(split_pools_.back(), &ed, &ev));
  split_events_.push_back(ev);
  ZE_CHECK(zeCommandListAppendBarrier(list_, ev, 0, nullptr));
  ZE_CHECK(zeCommandListClose(list_));
  ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  ld.commandQueueGroupOrdinal = ordinal_;
  ld.flags = in_order_ ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0;
  ZE_CHECK(zeCommandListCreate(rt_.context(), rt_.cards().at(card_).device, &ld, &list_));
  ZE_CHECK(zeCommandListAppendBarrier(list_, nullptr, 1, &ev));
  ze_fence_desc_t fd = {ZE_STRUCTURE_TYPE_FENCE_DESC};
  ZE_CHECK(zeFenceCreate(queue_, &fd, &fence_));
  parts_.push_back({list_, fence_, false});
  together_ = false;
}

void List::close() {
  ONE_USER;
  if (kind_ == regular) ZE_CHECK(zeCommandListClose(list_));
}

void List::execute() {
  ONE_USER;
  if (kind_ != regular) throw std::runtime_error("execute() on an immediate list");
  execute_part(0);
}

void List::execute_part(uint32_t part) {
  ONE_USER;
  if (kind_ != regular) throw std::runtime_error("execute_part() on an immediate list");
  if (part == 0)
    for (ze_event_handle_t e : split_events_) ZE_CHECK(zeEventHostReset(e));
  Part &p = parts_.at(part);
  if (p.submitted) ZE_CHECK(zeFenceReset(p.fence));
  ZE_CHECK(zeCommandQueueExecuteCommandLists(queue_, 1, &p.list, p.fence));
  p.submitted = true;
  if (part == 0) submitted_ = true;
}

bool Runtime::peer_access(uint32_t a, uint32_t b, Library *lib, std::string *why) {
  (void)cards_.at(a);
  (void)cards_.at(b);
  if (a == b) return true;
  auto fail = [&](const std::string &msg) {
    if (why) *why = msg;
    return false;
  };
  for (int dir = 0; dir < 2; dir++) {
    const uint32_t s = dir ? b : a, d = dir ? a : b;
    const ze_device_handle_t ds = cards_[s].device, dd = cards_[d].device;
    ze_bool_t can = 0;
    ze_result_t r = zeDeviceCanAccessPeer(ds, dd, &can);
    if (!check_peer_probe(r, "zeDeviceCanAccessPeer", why)) return false;
    if (!can) return fail("zeDeviceCanAccessPeer is false for card " + std::to_string(s) + " -> " + std::to_string(d));
    ze_device_p2p_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES};
    r = zeDeviceGetP2PProperties(ds, dd, &p);
    if (!check_peer_probe(r, "zeDeviceGetP2PProperties", why)) return false;
    if (!(p.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS))
      return fail("zeDeviceGetP2PProperties has no access flag for card " + std::to_string(s) + " -> " +
                  std::to_string(d));
  }
  // Test kernel peer stores before peer blitter copies: boards where peer EU stores are dropped fail cleanly here
  // instead of triggering a copy-engine BAR-to-BAR DMA fault.
  std::unique_ptr<Library> local_lib;
  if (!lib) {
    local_lib = std::make_unique<Library>(*this);
    lib = local_lib.get();
  }
  constexpr size_t kBytes = 4096;
  constexpr uint32_t kWords = kBytes / sizeof(uint32_t);
  static_assert(kWords % 64 == 0);
  struct Guard {
    Runtime *rt;
    void *p[4] = {};
    ~Guard() {
      if (stuck) return;
      for (void *x : p)
        if (x) rt->free(x);
    }
  } guard{this};
  auto *hsrc = static_cast<uint8_t *>(guard.p[0] = alloc_host(kBytes));
  auto *hdst = static_cast<uint8_t *>(guard.p[1] = alloc_host(kBytes));
  void *pa = guard.p[2] = alloc_device(a, kBytes);
  void *pb = guard.p[3] = alloc_device(b, kBytes);
  for (size_t i = 0; i < kBytes; i++) hsrc[i] = static_cast<uint8_t>(i * 131u + 0x5au);
  const uint8_t sentinel = 0x33;
  auto run_list = [&](List &peer, const std::string &tag) -> bool {
    ze_result_t r = zeCommandListClose(peer.list_);
    if (!check_peer_probe(r, tag + "zeCommandListClose", why)) return false;
    r = zeCommandQueueExecuteCommandLists(peer.queue_, 1, &peer.list_, peer.fence_);
    if (!check_peer_probe(r, tag + "zeCommandQueueExecuteCommandLists", why)) return false;
    peer.submitted_ = true;
    peer.parts_[0].submitted = true;
    r = zeFenceHostSynchronize(peer.fence_, kListWaitNs);
    if (!check_peer_probe(r, tag + "zeFenceHostSynchronize", why, true)) return false;
    return true;
  };
  for (int dir = 0; dir < 2; dir++) {
    const uint32_t s = dir ? b : a, d = dir ? a : b;
    void *pd = dir ? pa : pb;
    const std::string tag = "peer kernel write card " + std::to_string(s) + " -> " + std::to_string(d) + ": ";
    List up_d(*this, d, List::immediate);
    up_d.fill(pd, &sentinel, 1, kBytes);
    up_d.synchronize();
    const ze_kernel_handle_t k = lib->get(s, "smoke", "smoke");
    const uint32_t bias = 0x5a0000u + (dir ? 0x1000u : 0u);
    ze_result_t r = zeKernelSetArgumentValue(k, 0, sizeof(pd), &pd);
    if (!check_peer_probe(r, tag + "zeKernelSetArgumentValue(0)", why)) return false;
    r = zeKernelSetArgumentValue(k, 1, sizeof(bias), &bias);
    if (!check_peer_probe(r, tag + "zeKernelSetArgumentValue(1)", why)) return false;
    r = zeKernelSetGroupSize(k, 64, 1, 1);
    if (!check_peer_probe(r, tag + "zeKernelSetGroupSize", why)) return false;
    List peer(*this, s, List::regular);
    const ze_group_count_t g = {kWords / 64, 1, 1};
    r = zeCommandListAppendLaunchKernel(peer.list_, k, &g, nullptr, 0, nullptr);
    if (!check_peer_probe(r, tag + "zeCommandListAppendLaunchKernel", why)) return false;
    if (!run_list(peer, tag)) return false;
    std::memset(hdst, 0xee, kBytes);
    up_d.copy(hdst, pd, kBytes);
    up_d.synchronize();
    const auto *got = reinterpret_cast<const uint32_t *>(hdst);
    for (uint32_t i = 0; i < kWords; i++)
      if (got[i] != i + bias) return fail(tag + "read back wrong values");
  }
  for (int dir = 0; dir < 2; dir++) {
    const uint32_t s = dir ? b : a, d = dir ? a : b;
    void *ps = dir ? pb : pa, *pd = dir ? pa : pb;
    const std::string tag = "peer copy card " + std::to_string(s) + " -> " + std::to_string(d) + ": ";
    List up_s(*this, s, List::immediate), up_d(*this, d, List::immediate);
    up_s.copy(ps, hsrc, kBytes);
    up_s.synchronize();
    up_d.fill(pd, &sentinel, 1, kBytes);
    up_d.synchronize();
    List peer(*this, s, List::regular);
    const ze_result_t r = zeCommandListAppendMemoryCopy(peer.list_, pd, ps, kBytes, nullptr, 0, nullptr);
    if (!check_peer_probe(r, tag + "zeCommandListAppendMemoryCopy", why)) return false;
    if (!run_list(peer, tag)) return false;
    std::memset(hdst, 0xee, kBytes);
    up_d.copy(hdst, pd, kBytes);
    up_d.synchronize();
    if (std::memcmp(hsrc, hdst, kBytes) != 0) return fail(tag + "read back wrong bytes");
  }
  return true;
}

void List::synchronize() {
  ONE_USER;
  if (kind_ == immediate) {
    const ze_result_t r = zeCommandListHostSynchronize(list_, kListWaitNs);
    if (r == ZE_RESULT_NOT_READY) {
      stuck = true;
      throw std::runtime_error("card " + std::to_string(card_) + " did not finish a list in " +
                               std::to_string(kListWaitNs / 1'000'000'000ull) +
                               " s; its engine may have been reset (the kernel log, /sys/class/devcoredump)");
    }
    check(r, "zeCommandListHostSynchronize");
    return;
  }
  synchronize_part(0);
}

void List::synchronize_part(uint32_t part) {
  ONE_USER;
  Part &p = parts_.at(part);
  if (!p.submitted) return;
  const ze_result_t r = zeFenceHostSynchronize(p.fence, kListWaitNs);
  if (r == ZE_RESULT_NOT_READY) {
    stuck = true;
    throw std::runtime_error("card " + std::to_string(card_) + " did not finish a list in " +
                             std::to_string(kListWaitNs / 1'000'000'000ull) +
                             " s; its engine may have been reset (the kernel log, /sys/class/devcoredump)");
  }
  check(r, "zeFenceHostSynchronize");
}

void List::reset_list() {
  ONE_USER;
  if (kind_ == regular) {
    for (size_t i = 1; i < parts_.size(); i++) {
      if (parts_[i].fence) zeFenceDestroy(parts_[i].fence);
      if (parts_[i].list) zeCommandListDestroy(parts_[i].list);
    }
    parts_.resize(1);
    for (ze_event_handle_t e : split_events_) zeEventDestroy(e);
    split_events_.clear();
    for (ze_event_pool_handle_t pool : split_pools_) zeEventPoolDestroy(pool);
    split_pools_.clear();
    Part &p = parts_[0];
    if (p.submitted) {
      ZE_CHECK(zeFenceReset(p.fence));
      p.submitted = false;
    }
    ZE_CHECK(zeCommandListReset(p.list));
    list_ = p.list;
    fence_ = p.fence;
    submitted_ = false;
    together_ = false;
  }
}

}  // namespace engine::ze
