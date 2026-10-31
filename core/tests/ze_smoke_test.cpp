// A kernel built by ocloc runs on every card, and a card waits on an event another card signals: the form of the
// exchanges between the cards. The lists are regular (recorded) lists, submitted in reverse order so that each
// wait is submitted before its signal; an immediate list blocks the host in that case. ENGINE_EXPECT_GPUS=N makes
// it fail unless N cards ran.
#include <cstdlib>
#include <memory>

#include "test_util.h"
#include "ze/ze.h"

using namespace engine;

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  const uint32_t n = 1024;
  std::vector<std::unique_ptr<ze::Module>> mods;
  std::vector<std::unique_ptr<ze::List>> lists;
  ze::EventPool events(rt, rt.ncards());
  auto *buf = static_cast<uint32_t *>(rt.alloc_host(n * sizeof(uint32_t) * rt.ncards()));
  for (uint32_t i = 0; i < n * rt.ncards(); i++) buf[i] = 0xffffffffu;
  for (uint32_t c = 0; c < rt.ncards(); c++) {
    const auto &card = rt.cards()[c];
    std::printf("card %u: %s at %s, %.2f GiB, shared system USM 0x%x\n", c, card.name.c_str(), card.pci.c_str(),
                card.memory / 1073741824.0, card.shared_system);
    mods.push_back(std::make_unique<ze::Module>(rt, c, ENGINE_KERNEL_DIR "/" + card.ip() + "/smoke.bin"));
    lists.push_back(std::make_unique<ze::List>(rt, c, ze::List::regular));
  }
  // Card c writes its slice with bias c; card c > 0 first waits for card c - 1's event.
  for (uint32_t c = rt.ncards(); c-- > 0;) {
    uint32_t *slice = buf + c * n;
    uint32_t bias = 1000 * c;
    auto k = mods[c]->kernel("smoke");
    if (c > 0) lists[c]->wait({events[c - 1]});
    lists[c]->launch(k, {slice, bias}, n / 64, 1, 1, 64, 1, 1, events[c]);
    lists[c]->close();
  }
  for (uint32_t c = rt.ncards(); c-- > 0;) lists[c]->execute();
  for (auto &l : lists) l->synchronize();
  uint32_t bad = 0;
  for (uint32_t c = 0; c < rt.ncards(); c++)
    for (uint32_t i = 0; i < n; i++) bad += buf[c * n + i] != i + 1000 * c;
  EXPECT(bad == 0, "%u values wrong", bad);

  // copy_rows on each card's copy engine, in the prompt path's shape: rows of a wide source pitch into a narrower one.
  {
    const size_t rows = 5, width = 300, sp = 1000, dp = 512;
    auto *src = static_cast<uint8_t *>(rt.alloc_host(rows * sp));
    auto *back = static_cast<uint8_t *>(rt.alloc_host(rows * dp));
    for (size_t i = 0; i < rows * sp; i++) src[i] = (uint8_t)(i * 7 + 3);
    for (uint32_t c = 0; c < rt.ncards(); c++) {
      auto *dev = static_cast<uint8_t *>(rt.alloc_device(c, rows * dp));
      const uint8_t zero = 0;
      ze::List L(rt, c, ze::List::immediate, true);
      L.fill(dev, &zero, 1, rows * dp);
      L.copy_rows(dev, dp, src + 7, sp, width, rows);
      L.copy(back, dev, rows * dp);
      L.synchronize();
      uint32_t wrong = 0;
      for (size_t r = 0; r < rows; r++)
        for (size_t i = 0; i < dp; i++) wrong += back[r * dp + i] != (i < width ? src[7 + r * sp + i] : 0);
      EXPECT(wrong == 0, "card %u: copy_rows, %u bytes wrong", c, wrong);
      rt.free(dev);
    }
    rt.free(src);
    rt.free(back);
  }
  if (const char *e = std::getenv("ENGINE_EXPECT_GPUS"))
    EXPECT(rt.ncards() == (uint32_t)std::atoi(e), "%u cards ran, ENGINE_EXPECT_GPUS=%s", rt.ncards(), e);
  std::printf("PASS: %u card(s)\n", rt.ncards());
  return 0;
}
