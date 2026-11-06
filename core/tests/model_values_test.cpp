// Level 1 (docs/TESTING.md): what the C ABI's code takes from a model instead of Flash-Next's values (api/model.h,
// ADR-0061): qwen4exp's image shape and draft counts, and the image arithmetic on that shape and on one without
// video. No card.
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "api/model.h"
#include "models/qwen4exp/model.h"
#include "test_util.h"

using engine::image_block_tokens;
using engine::image_bytes;
using engine::ImageShape;
using engine::is_media;
using engine::kNoToken;

int main() {
  // The tokens are the file's, found by their text; a file without video has no video token.
  std::map<std::string, engine::GgufValue> meta;
  meta["tokenizer.ggml.tokens"] = std::vector<std::string>{"a", "<|image_pad|>", "b", "<|video_pad|>"};
  const ImageShape f = engine::qwen4exp::image_shape_of(meta);
  EXPECT(f.image_token == 1 && f.video_token == 3, "the file's image and video tokens");
  EXPECT(f.patch_bytes == 768 && f.merge == 2, "16-pixel RGB patches merged 2 x 2");
  meta["tokenizer.ggml.tokens"] = std::vector<std::string>{"a", "<|image_pad|>"};
  EXPECT(engine::qwen4exp::image_shape_of(meta).video_token == kNoToken, "no video token");

  // Flash-Next's file: <|image_pad|> 248,056 and <|video_pad|> 248,057.
  const ImageShape q{248056, 248057, f.patch_bytes, f.merge};
  EXPECT(engine::qwen4exp::kDraftMaxLayer == 3, "the draft layer's default drafts a round");
  EXPECT(engine::qwen4exp::kDraftMaxCap == 7, "the most drafts a round: a window's rows but one");

  // Flash-Next's image of 32 x 48 patches: 16 x 24 tokens, each 2 x 2 patches of 768 bytes, once a frame.
  EXPECT(image_block_tokens(q, 1, 32, 48) == 384, "an image's tokens");
  EXPECT(image_bytes(q, 384, 1) == 384ull * 4 * 768, "an image's bytes");
  EXPECT(image_block_tokens(q, 2, 32, 48) == 384 && image_bytes(q, 384, 2) == 384ull * 4 * 768 * 2,
         "a pair of video frames");
  EXPECT(image_block_tokens(q, 1, 31, 48) == 0 && image_block_tokens(q, 1, 32, 47) == 0, "a side not merged whole");
  EXPECT(image_block_tokens(q, 3, 32, 48) == 0 && image_block_tokens(q, 0, 32, 48) == 0, "neither 1 nor 2 frames");
  EXPECT(is_media(q, 248056) && is_media(q, 248057) && !is_media(q, 248055), "Flash-Next's media tokens");
  EXPECT(!is_media(q, kNoToken), "no token is not media");

  // A shape without video and without merging: 2 frames are refused, and kNoToken is not its video token.
  const ImageShape s{1000, kNoToken, 588, 1};
  EXPECT(image_block_tokens(s, 1, 3, 5) == 15 && image_bytes(s, 15, 1) == 15ull * 588, "no merge");
  EXPECT(image_block_tokens(s, 2, 4, 4) == 0, "no video token: no pair of frames");
  EXPECT(is_media(s, 1000) && !is_media(s, kNoToken), "only the image token is media");

  std::printf("model_values_test: ok\n");
  return 0;
}
