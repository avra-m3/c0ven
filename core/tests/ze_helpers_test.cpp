// Host-only checks for Level Zero chip-family matching, SPIR-V directory selection, and peer-probe error handling.
#include <stdexcept>
#include <string>

#include "test_util.h"
#include "ze/library.h"
#include "ze/ze.h"

using namespace engine;

namespace {

constexpr uint32_t make_ip(uint32_t major, uint32_t rel, uint32_t rev) {
  return (major << 22) | (rel << 14) | rev;
}

ze::Card card_with_ip(const char *name, uint32_t ip) {
  ze::Card c;
  c.name = name;
  c.ip_version = ip;
  return c;
}

}  // namespace

int main() {
  EXPECT(!card_with_ip("none", 0).is_hpg(), "unreported ip must not match HPG");
  EXPECT(!card_with_ip("TGL", make_ip(12, 0, 0)).is_hpg(), "Tiger Lake (12.0.0) must not match HPG");
  EXPECT(!card_with_ip("DG1", make_ip(12, 10, 0)).is_hpg(), "DG1 (12.10.0) must not match HPG");
  EXPECT(!card_with_ip("RPL", make_ip(12, 30, 0)).is_hpg(), "RPL (12.30.0) must not match HPG");
  EXPECT(!card_with_ip("PVC", make_ip(12, 60, 7)).is_hpg(), "Ponte Vecchio (12.60.7) must not match HPG");
  EXPECT(!card_with_ip("MTL", make_ip(12, 70, 4)).is_hpg(), "Meteor Lake (12.70.4) must not match HPG");
  EXPECT(!card_with_ip("B60", make_ip(20, 1, 0)).is_hpg(), "Battlemage G21 (20.1.0) must not match HPG");
  EXPECT(!card_with_ip("B580", make_ip(20, 2, 0)).is_hpg(), "Battlemage G31 (20.2.0) must not match HPG");
  for (const uint32_t ip : {make_ip(12, 55, 0), make_ip(12, 55, 8), make_ip(12, 56, 5), make_ip(12, 57, 0)})
    EXPECT(card_with_ip("ACM", ip).is_hpg(), "DG2 IP 0x%x must match HPG", ip);

  const ze::Card acm = card_with_ip("Arc A770", make_ip(12, 55, 8));
  const ze::Card bmg = card_with_ip("Arc Pro B60", make_ip(20, 1, 0));
  EXPECT(ze::select_spirv_dir(acm, "/k/20.1.0", "/k/12.55.8") == "/k/12.55.8", "ACM selects HPG SPIR-V");
  EXPECT(ze::select_spirv_dir(bmg, "/k/20.1.0", "/k/12.55.8") == "/k/20.1.0", "BMG selects Xe2 SPIR-V");

  for (bool hpg_card : {false, true}) {
    bool threw = false;
    try {
      (void)ze::select_spirv_dir(hpg_card ? acm : bmg, hpg_card ? "/k/20.1.0" : "", hpg_card ? "" : "/k/12.55.8");
    } catch (const std::runtime_error &e) {
      threw = std::string(e.what()).find(hpg_card ? "Xe-HPG" : "Xe2") != std::string::npos;
    }
    EXPECT(threw, "missing %s SPIR-V must throw naming family", hpg_card ? "Xe-HPG" : "Xe2");
  }

  std::string why;
  EXPECT(ze::check_peer_probe(ZE_RESULT_SUCCESS, "probe", &why), "success passes peer probe");
  EXPECT(!ze::queue_stuck(), "success leaves queue_stuck false");
  EXPECT(!ze::check_peer_probe(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, "probe", &why), "unsupported fails cleanly");
  EXPECT(why.find("0x") != std::string::npos, "why explains non-fatal peer probe failure");
  EXPECT(!ze::queue_stuck(), "non-lost peer probe error leaves queue_stuck false");

  bool cleanup_ran = false;
  bool lost_threw = false;
  try {
    struct Guard {
      bool &ran;
      ~Guard() {
        if (ze::queue_stuck()) return;
        ran = true;
      }
    } guard{cleanup_ran};
    (void)ze::check_peer_probe(ZE_RESULT_ERROR_DEVICE_LOST, "peer kernel write card 0 -> 1", &why);
  } catch (const std::runtime_error &) {
    lost_threw = true;
  }
  EXPECT(lost_threw, "DEVICE_LOST in peer probe throws");
  EXPECT(ze::queue_stuck(), "DEVICE_LOST in peer probe marks queue_stuck");
  EXPECT(!cleanup_ran, "DEVICE_LOST prevents peer-probe cleanup on the lost context");
  return 0;
}
