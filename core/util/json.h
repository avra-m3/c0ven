// A small JSON reader for the files the core reads as published: a safetensors header and a drafter's
// config.json (docs/MODEL_FILES.md). Numbers are doubles; a \uXXXX escape past ASCII is refused.
#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace engine {

struct Json {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0;
  std::string string;
  std::vector<Json> array;
  std::map<std::string, Json> object;

  // The member key of an object; throws naming `key` when this is not an object or has no such member.
  const Json &at(const std::string &key) const;
  // A member, or null when this is not an object or has no such member.
  const Json *find(const std::string &key) const;
};

// The one JSON value in text, with nothing but white space around it. Throws std::runtime_error naming the offset
// of the first byte that does not fit.
Json parse_json(std::string_view text);

}  // namespace engine
