#include "ze.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace engine::ze {

void check(ze_result_t r, const char *what) {
  if (r == ZE_RESULT_SUCCESS) return;
  char buf[64];
  std::snprintf(buf, sizeof buf, " returned 0x%x", (unsigned)r);
  throw std::runtime_error(std::string(what) + buf);
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
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
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2};
      ZE_CHECK(zeDeviceGetProperties(dev, &p));
      if (p.type != ZE_DEVICE_TYPE_GPU) continue;
      Card c;
      c.device = dev;
      c.name = p.name;
      c.max_alloc = p.maxMemAllocSize;
      c.timer_hz = p.timerResolution;  // cycles per second, with the 1.2 properties
      c.timestamp_mask = p.timestampValidBits >= 64 ? ~0ull : ((1ull << p.timestampValidBits) - 1);
      uint32_t nm = 0;
      ZE_CHECK(zeDeviceGetMemoryProperties(dev, &nm, nullptr));
      std::vector<ze_device_memory_properties_t> mp(nm, {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES});
      ZE_CHECK(zeDeviceGetMemoryProperties(dev, &nm, mp.data()));
      for (auto &m : mp) c.memory += m.totalSize;
      ze_device_memory_access_properties_t ma = {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
      ZE_CHECK(zeDeviceGetMemoryAccessProperties(dev, &ma));
      c.shared_system = ma.sharedSystemAllocCapabilities;
      ze_pci_ext_properties_t pci = {ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
      if (zeDevicePciGetPropertiesExt(dev, &pci) == ZE_RESULT_SUCCESS) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x", pci.address.domain, pci.address.bus, pci.address.device,
                      pci.address.function);
        c.pci = buf;
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
    ZE_CHECK(zeMemFree(ctx_, it->first));
    device_bytes_[card] -= it->second.second;
    it = allocs_.erase(it);
  }
}

Module::Module(Runtime &rt, uint32_t card, const std::string &path) {
  auto bin = read_file(path);
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC};
  md.format = ZE_MODULE_FORMAT_NATIVE;
  md.inputSize = bin.size();
  md.pInputModule = bin.data();
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

Module::~Module() {
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
  for (auto e : events_) zeEventDestroy(e);
  if (pool_) zeEventPoolDestroy(pool_);
}

List::List(Runtime &rt, uint32_t card, Kind kind, bool copy_engine, bool in_order)
    : rt_(rt), card_(card), kind_(kind), in_order_(in_order) {
  const Card &c = rt.cards().at(card);
  uint32_t ordinal = copy_engine && c.copy_ordinal != UINT32_MAX ? c.copy_ordinal : c.compute_ordinal;
  ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = ordinal;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  qd.flags = in_order || kind == regular ? ZE_COMMAND_QUEUE_FLAG_IN_ORDER : 0;
  if (kind == immediate) {
    ZE_CHECK(zeCommandListCreateImmediate(rt.context(), c.device, &qd, &list_));
  } else {
    ZE_CHECK(zeCommandQueueCreate(rt.context(), c.device, &qd, &queue_));
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ld.commandQueueGroupOrdinal = ordinal;
    ld.flags = in_order ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0;
    ZE_CHECK(zeCommandListCreate(rt.context(), c.device, &ld, &list_));
    ze_fence_desc_t fd = {ZE_STRUCTURE_TYPE_FENCE_DESC};
    ZE_CHECK(zeFenceCreate(queue_, &fd, &fence_));
  }
}

List::~List() {
  if (fence_) zeFenceDestroy(fence_);
  if (list_) zeCommandListDestroy(list_);
  if (queue_) zeCommandQueueDestroy(queue_);
}

void List::launch(ze_kernel_handle_t k, std::initializer_list<Arg> args, uint32_t gx, uint32_t gy, uint32_t gz,
                  uint32_t lx, uint32_t ly, uint32_t lz, ze_event_handle_t signal,
                  std::initializer_list<ze_event_handle_t> wait) {
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
  trace_ = ts;
  trace_cap_ = cap;
  trace_names_.clear();
  timestamp(ts);
}

void List::copy(void *dst, const void *src, size_t bytes, ze_event_handle_t signal,
                std::initializer_list<ze_event_handle_t> wait) {
  std::vector<ze_event_handle_t> w(wait);
  ZE_CHECK(zeCommandListAppendMemoryCopy(list_, dst, src, bytes, signal, (uint32_t)w.size(),
                                         w.empty() ? nullptr : w.data()));
  if (!in_order_) barrier();
}

void List::copy_rows(void *dst, size_t dst_pitch, const void *src, size_t src_pitch, size_t width, size_t rows) {
  const ze_copy_region_t region = {0, 0, 0, (uint32_t)width, (uint32_t)rows, 1};
  ZE_CHECK(zeCommandListAppendMemoryCopyRegion(list_, dst, &region, (uint32_t)dst_pitch, 0, src, &region,
                                               (uint32_t)src_pitch, 0, nullptr, 0, nullptr));
  if (!in_order_) barrier();
}

void List::fill(void *dst, const void *pattern, size_t pattern_size, size_t bytes) {
  ZE_CHECK(zeCommandListAppendMemoryFill(list_, dst, pattern, pattern_size, bytes, nullptr, 0, nullptr));
  if (!in_order_) barrier();
}

void List::wait(std::initializer_list<ze_event_handle_t> events) {
  std::vector<ze_event_handle_t> w(events);
  ZE_CHECK(zeCommandListAppendWaitOnEvents(list_, (uint32_t)w.size(), w.data()));
}

void List::signal(ze_event_handle_t e) { ZE_CHECK(zeCommandListAppendSignalEvent(list_, e)); }
void List::reset(ze_event_handle_t e) { ZE_CHECK(zeCommandListAppendEventReset(list_, e)); }
void List::barrier() { ZE_CHECK(zeCommandListAppendBarrier(list_, nullptr, 0, nullptr)); }

void List::together(bool on) {
  if (in_order_) return;
  if (!on && together_ && !trace_) barrier();  // while tracing, each launch already had its own
  together_ = on;
}

void List::timestamp(uint64_t *dst) {
  ZE_CHECK(zeCommandListAppendWriteGlobalTimestamp(list_, dst, nullptr, 0, nullptr));
}

void List::close() {
  if (kind_ == regular) ZE_CHECK(zeCommandListClose(list_));
}

void List::execute() {
  if (kind_ != regular) throw std::runtime_error("execute() on an immediate list");
  if (submitted_) ZE_CHECK(zeFenceReset(fence_));
  ZE_CHECK(zeCommandQueueExecuteCommandLists(queue_, 1, &list_, fence_));
  submitted_ = true;
}

// A list on a queue whose engine the driver reset never finishes, and the driver's job timeout does not apply to the
// engine's long-running queues, so an unbounded wait hangs the process. No list the engine records runs this long.
constexpr uint64_t kListWaitNs = 60'000'000'000ull;

void List::synchronize() {
  if (kind_ != immediate && !submitted_) return;
  const ze_result_t r = kind_ == immediate ? zeCommandListHostSynchronize(list_, kListWaitNs)
                                           : zeFenceHostSynchronize(fence_, kListWaitNs);
  if (r == ZE_RESULT_NOT_READY)
    throw std::runtime_error("card " + std::to_string(card_) + " did not finish a list in " +
                             std::to_string(kListWaitNs / 1'000'000'000ull) +
                             " s; its engine may have been reset (the kernel log, /sys/class/devcoredump)");
  check(r, kind_ == immediate ? "zeCommandListHostSynchronize" : "zeFenceHostSynchronize");
}

void List::reset_list() {
  if (kind_ == regular) ZE_CHECK(zeCommandListReset(list_));
}

}  // namespace engine::ze
