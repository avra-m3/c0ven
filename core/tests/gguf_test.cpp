// The GGUF reader against the IQ3_S file's census (docs/model/tensors-IQ3_S.tsv), and the
// O_DIRECT reader (load/direct.h) against the mapping: a tensor whole, an expert's slice, and the end of a shard.
#include <cstring>

#include "load/direct.h"
#include "load/gguf.h"
#include "quant.h"
#include "test_util.h"

#include <cstring>
#include <vector>

int main() {
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  engine::GgufModel m(path);
  EXPECT(m.files().size() == 2, "expected 2 shards, found %zu", m.files().size());
  EXPECT(m.tensors().size() == 1224, "expected 1,224 tensors, found %zu", m.tensors().size());
  EXPECT(m.get_string("general.architecture") == "qwen4exp", "architecture");
  EXPECT(m.get_int("qwen4exp.block_count") == 48, "block_count");
  EXPECT(m.get_int("qwen4exp.embedding_length") == 2560, "embedding_length");
  EXPECT(m.get_int("qwen4exp.expert_count") == 512, "expert_count");
  EXPECT(m.get_ints("qwen4exp.ple.head_vocab_sizes").size() == 16, "ple.head_vocab_sizes");
  EXPECT(m.get_ints("qwen4exp.ple.layer_multipliers")[0] == 23703573157769LL, "ple multiplier 0");

  const auto &out = m.tensor("output.weight");
  EXPECT(out.type == QT_Q6_K && out.shape[0] == 2560 && out.shape[1] == 248320, "output.weight shape");
  EXPECT(out.shard == 0 && out.offset == 11024384 && out.bytes == 521472000, "output.weight placement");
  const auto &ng = m.tensor("per_layer_token_embd.weight");
  EXPECT(ng.shard == 1 && ng.offset == 192 && ng.bytes == 28800138240ULL, "n-gram table placement");
  const auto &ex = m.tensor("blk.0.ffn_down_exps.weight");
  EXPECT(ex.shape.size() == 3 && ex.shape[0] == 640 && ex.shape[1] == 2560 && ex.shape[2] == 512, "expert shape");

  // Releasing one expert's range of a tensor (a swap's read) leaves its bytes readable and the same.
  {
    const uint64_t ed = ex.bytes / 512, off = 3 * ed;
    std::vector<uint8_t> before(ex.data + off, ex.data + off + ed);
    m.release(ex, off, ed);
    EXPECT(std::memcmp(before.data(), ex.data + off, ed) == 0, "expert 3 of blk.0.ffn_down_exps changed after release");
  }

  uint64_t total = 0;
  const engine::GgufTensor *last = nullptr;  // the tensor that ends shard 0
  for (auto &[name, t] : m.tensors()) {
    total += t.bytes;
    if (t.shard == 0 && (!last || t.offset > last->offset)) last = &t;
  }
  EXPECT(total == 83606638080ULL, "total tensor bytes %llu", (unsigned long long)total);

  engine::DirectReader reader(m.files());
  auto same = [&](const engine::GgufTensor &t, uint64_t off, uint64_t bytes, bool whole) {
    auto buf = engine::aligned_buffer(engine::DirectReader::span(t.offset + off, bytes));
    const uint8_t *p = whole ? reader.read(t.shard, t.offset + off, bytes, buf.get())
                             : reader.read_one(t.shard, t.offset + off, bytes, buf.get());
    bool ok = std::memcmp(p, t.data + off, bytes) == 0;
    m.release(t);
    return ok;
  };
  const auto &qkv = m.tensor("blk.0.attn_qkv.weight");
  EXPECT(same(qkv, 0, qkv.bytes, true), "blk.0.attn_qkv.weight read with O_DIRECT differs from the mapping");
  const uint64_t ed = ex.bytes / 512;
  EXPECT(same(ex, 7 * ed, ed, false), "expert 7 of blk.0.ffn_down_exps.weight differs");
  EXPECT(same(*last, last->bytes - 1000, 1000, true), "the end of %s differs", last->name.c_str());
  std::printf("PASS: 1,224 tensors, %llu bytes; O_DIRECT reads equal the mapping\n", (unsigned long long)total);
  return 0;
}
