#include "util/json.h"

#include <cstdlib>
#include <stdexcept>

namespace engine {

namespace {

struct Reader {
  std::string_view s;
  size_t i = 0;

  [[noreturn]] void fail(const char *what) const {
    throw std::runtime_error(std::string("JSON: ") + what + " at byte " + std::to_string(i));
  }
  void space() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
  }
  bool take(char c) {
    space();
    if (i < s.size() && s[i] == c) {
      i++;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!take(c)) fail("unexpected character");
  }
  bool word(std::string_view w) {
    if (s.substr(i, w.size()) != w) return false;
    i += w.size();
    return true;
  }

  std::string str() {
    expect('"');
    std::string out;
    while (true) {
      if (i >= s.size()) fail("unterminated string");
      char c = s[i++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i >= s.size()) fail("unterminated string");
      switch (char e = s[i++]) {
        case '"': case '\\': case '/': out += e; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (i + 4 > s.size()) fail("short \\u escape");
          unsigned long v = std::strtoul(std::string(s.substr(i, 4)).c_str(), nullptr, 16);
          if (v >= 0x80) fail("a \\u escape past ASCII");
          out += (char)v;
          i += 4;
          break;
        }
        default: fail("unknown escape");
      }
    }
  }

  Json value(int depth) {
    if (depth > 64) fail("nested too deep");
    space();
    if (i >= s.size()) fail("missing value");
    Json v;
    char c = s[i];
    if (c == '{') {
      i++;
      v.kind = Json::Kind::Object;
      if (take('}')) return v;
      do {
        space();
        std::string key = str();
        expect(':');
        v.object[key] = value(depth + 1);
      } while (take(','));
      expect('}');
    } else if (c == '[') {
      i++;
      v.kind = Json::Kind::Array;
      if (take(']')) return v;
      do v.array.push_back(value(depth + 1));
      while (take(','));
      expect(']');
    } else if (c == '"') {
      v.kind = Json::Kind::String;
      v.string = str();
    } else if (word("true")) {
      v.kind = Json::Kind::Bool;
      v.boolean = true;
    } else if (word("false")) {
      v.kind = Json::Kind::Bool;
    } else if (word("null")) {
    } else {
      const std::string rest(s.substr(i, 64));
      char *end = nullptr;
      v.number = std::strtod(rest.c_str(), &end);
      if (end == rest.c_str()) fail("not a value");
      v.kind = Json::Kind::Number;
      i += (size_t)(end - rest.c_str());
    }
    return v;
  }
};

}  // namespace

const Json *Json::find(const std::string &key) const {
  if (kind != Kind::Object) return nullptr;
  auto it = object.find(key);
  return it == object.end() ? nullptr : &it->second;
}

const Json &Json::at(const std::string &key) const {
  const Json *v = find(key);
  if (!v) throw std::runtime_error("JSON: no member " + key);
  return *v;
}

Json parse_json(std::string_view text) {
  Reader r{text};
  Json v = r.value(0);
  r.space();
  if (r.i != text.size()) r.fail("text after the value");
  return v;
}

}  // namespace engine
