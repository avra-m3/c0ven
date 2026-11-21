// Streaming hidden-state probe loader, CPU reference forward pass, and per-slot rolling-mean and drift tracker.
#include "api/probe.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/json.h"

namespace engine {

namespace {

inline float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 64);
  return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t frac = h & 0x3ffu;
  uint32_t u = 0;
  if (exp == 0) {
    if (frac == 0) {
      u = sign;
    } else {
      exp = 1;
      while ((frac & 0x400u) == 0) {
        frac <<= 1;
        exp--;
      }
      frac &= 0x3ffu;
      u = sign | ((exp + 112u) << 23) | (frac << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7f800000u | (frac << 13);
  } else {
    u = sign | ((exp + 112u) << 23) | (frac << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline float sigmoid_f(float x) {
  if (x >= 0.0f) {
    const float e = std::exp(-x);
    return 1.0f / (1.0f + e);
  }
  const float e = std::exp(x);
  return e / (1.0f + e);
}

inline float silu_f(float x) { return x * sigmoid_f(x); }

void copy_name(char dst[ENGINE_MAX_PROBE_NAME], const std::string &src) {
  std::memset(dst, 0, ENGINE_MAX_PROBE_NAME);
  const size_t n = std::min(src.size(), (size_t)(ENGINE_MAX_PROBE_NAME - 1));
  std::memcpy(dst, src.data(), n);
}

uint32_t parse_scope(const std::string &s, const std::string &where) {
  if (s == "input") return ENGINE_PROBE_SCOPE_INPUT;
  if (s == "output") return ENGINE_PROBE_SCOPE_OUTPUT;
  if (s == "both") return ENGINE_PROBE_SCOPE_BOTH;
  if (s == "drift") return ENGINE_PROBE_SCOPE_DRIFT;
  throw std::runtime_error(where + ": unknown probe scope '" + s + "'");
}

std::string read_text_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

// Safetensors reader supporting BF16, F16, and F32 tensors.
struct SafeFile {
  struct Fd {
    int v = -1;
    ~Fd() {
      if (v >= 0) ::close(v);
    }
  };
  struct Entry {
    uint64_t begin = 0, end = 0;
    std::string dtype;
    std::vector<uint64_t> shape;
  };
  std::string path;
  Fd fd;
  std::map<std::string, Entry> entries;

  explicit SafeFile(const std::string &p) : path(p), fd{::open(p.c_str(), O_RDONLY | O_CLOEXEC)} {
    if (fd.v < 0) throw std::runtime_error("cannot open " + path);
    uint64_t hlen = 0;
    if (::pread(fd.v, &hlen, 8, 0) != 8 || hlen == 0 || hlen > (64u << 20)) fail("invalid safetensors header");
    std::string text(hlen, '\0');
    if (::pread(fd.v, text.data(), hlen, 8) != (ssize_t)hlen) fail("short safetensors header");
    const Json header = parse_json(text);
    if (header.kind != Json::Kind::Object) fail("safetensors header is not an object");
    for (const auto &[name, val] : header.object) {
      if (name == "__metadata__") continue;
      Entry e;
      e.dtype = val.at("dtype").string;
      const Json &off = val.at("data_offsets");
      if (off.kind != Json::Kind::Array || off.array.size() != 2) fail(name + ": bad data_offsets");
      e.begin = 8 + hlen + (uint64_t)off.array[0].number;
      e.end = 8 + hlen + (uint64_t)off.array[1].number;
      const Json &sh = val.at("shape");
      if (sh.kind != Json::Kind::Array) fail(name + ": bad shape");
      for (const Json &d : sh.array) e.shape.push_back((uint64_t)d.number);
      entries[name] = std::move(e);
    }
  }

  [[noreturn]] void fail(const std::string &what) const { throw std::runtime_error(path + ": " + what); }

  bool has(const std::string &name) const { return entries.find(name) != entries.end(); }

  void read_bytes(const Entry &e, uint8_t *dst, size_t nbytes, const std::string &name) const {
    if (e.end < e.begin || e.end - e.begin != nbytes) fail(name + ": byte size mismatch");
    size_t got = 0;
    while (got < nbytes) {
      ssize_t r = ::pread(fd.v, dst + got, nbytes - got, (off_t)(e.begin + got));
      if (r <= 0) fail(name + ": short read");
      got += (size_t)r;
    }
  }

  void check_shape(const Entry &e, const std::vector<uint64_t> &want, const std::string &name) const {
    if (e.shape != want) fail(name + ": unexpected tensor shape");
  }

  std::vector<float> read_f32(const std::string &name, const std::vector<uint64_t> &want) const {
    auto it = entries.find(name);
    if (it == entries.end()) fail("missing tensor " + name);
    const Entry &e = it->second;
    check_shape(e, want, name);
    size_t el = 1;
    for (uint64_t d : want) el *= (size_t)d;
    std::vector<float> out(el);
    if (e.dtype == "F32") {
      read_bytes(e, reinterpret_cast<uint8_t *>(out.data()), el * 4, name);
    } else if (e.dtype == "BF16") {
      std::vector<uint16_t> raw(el);
      read_bytes(e, reinterpret_cast<uint8_t *>(raw.data()), el * 2, name);
      for (size_t i = 0; i < el; i++) out[i] = bf16_to_f32(raw[i]);
    } else if (e.dtype == "F16") {
      std::vector<uint16_t> raw(el);
      read_bytes(e, reinterpret_cast<uint8_t *>(raw.data()), el * 2, name);
      for (size_t i = 0; i < el; i++) out[i] = fp16_to_f32(raw[i]);
    } else {
      fail(name + ": unsupported dtype " + e.dtype);
    }
    return out;
  }

  std::vector<uint16_t> read_bf16(const std::string &name, const std::vector<uint64_t> &want) const {
    auto it = entries.find(name);
    if (it == entries.end()) fail("missing tensor " + name);
    const Entry &e = it->second;
    check_shape(e, want, name);
    size_t el = 1;
    for (uint64_t d : want) el *= (size_t)d;
    std::vector<uint16_t> out(el);
    if (e.dtype == "BF16") {
      read_bytes(e, reinterpret_cast<uint8_t *>(out.data()), el * 2, name);
    } else if (e.dtype == "F32") {
      std::vector<float> raw(el);
      read_bytes(e, reinterpret_cast<uint8_t *>(raw.data()), el * 4, name);
      for (size_t i = 0; i < el; i++) out[i] = f32_to_bf16(raw[i]);
    } else if (e.dtype == "F16") {
      std::vector<uint16_t> raw(el);
      read_bytes(e, reinterpret_cast<uint8_t *>(raw.data()), el * 2, name);
      for (size_t i = 0; i < el; i++) out[i] = f32_to_bf16(fp16_to_f32(raw[i]));
    } else {
      fail(name + ": unsupported dtype " + e.dtype);
    }
    return out;
  }
};

// Default label schema for the published 10-label Qwen3.8-Flash-Next SingProbe checkpoint.
std::vector<ProbeLabelSpec> default_singprobe_labels(uint32_t n) {
  static const struct {
    const char *name;
    uint32_t scope;
    uint32_t flags;
  } kDefaults[10] = {
      {"sexual", ENGINE_PROBE_SCOPE_INPUT, 0},
      {"crime_public_safety", ENGINE_PROBE_SCOPE_INPUT, 0},
      {"unethical_behavior", ENGINE_PROBE_SCOPE_INPUT, 0},
      {"cybersecurity_manipulation", ENGINE_PROBE_SCOPE_INPUT, 0},
      {"agent_safety", ENGINE_PROBE_SCOPE_INPUT, 0},
      {"politically_sensitive", ENGINE_PROBE_SCOPE_INPUT, 0},
      {"animal_abuse", ENGINE_PROBE_SCOPE_INPUT, 0},
      {"safe", ENGINE_PROBE_SCOPE_INPUT, ENGINE_PROBE_FLAG_SAFE},
      {"unsafe_response", ENGINE_PROBE_SCOPE_OUTPUT, 0},
      {"hallucination", ENGINE_PROBE_SCOPE_OUTPUT, 0},
  };
  std::vector<ProbeLabelSpec> out(n);
  for (uint32_t i = 0; i < n; i++) {
    if (n == 10) {
      out[i].name = kDefaults[i].name;
      out[i].scope = kDefaults[i].scope;
      out[i].flags = kDefaults[i].flags;
    } else {
      out[i].name = "label_" + std::to_string(i);
      out[i].scope = ENGINE_PROBE_SCOPE_BOTH;
    }
  }
  return out;
}

std::vector<uint32_t> read_layer_ids(const Json &cfg, const std::string &path, uint32_t model_layers) {
  const Json &arr = cfg.at("base_model_layer_ids");
  if (arr.kind != Json::Kind::Array || arr.array.empty() || arr.array.size() > 8)
    throw std::runtime_error(path + ": base_model_layer_ids must have 1..8 entries");
  std::vector<uint32_t> ids;
  for (size_t i = 0; i < arr.array.size(); i++) {
    if (arr.array[i].kind != Json::Kind::Number || arr.array[i].number < 0)
      throw std::runtime_error(path + ": invalid layer id");
    const uint32_t l = (uint32_t)arr.array[i].number;
    if (i > 0 && l <= ids.back()) throw std::runtime_error(path + ": base_model_layer_ids must strictly rise");
    if (model_layers > 0 && l >= model_layers)
      throw std::runtime_error(path + ": tapped layer " + std::to_string(l) + " >= model layers " +
                               std::to_string(model_layers));
    ids.push_back(l);
  }
  return ids;
}

void parse_c0ven_labels(const Json &cfg, const std::string &path, std::vector<ProbeLabelSpec> &out,
                        uint32_t base_linear_idx) {
  const Json &arr = cfg.at("labels");
  if (arr.kind != Json::Kind::Array || arr.array.empty())
    throw std::runtime_error(path + ": labels must be a non-empty array");
  for (size_t i = 0; i < arr.array.size(); i++) {
    const Json &item = arr.array[i];
    if (item.kind != Json::Kind::Object) throw std::runtime_error(path + ": labels[] entry must be an object");
    ProbeLabelSpec ls;
    ls.name = item.at("name").string;
    if (ls.name.empty()) throw std::runtime_error(path + ": empty label name");
    if (const Json *sc = item.find("scope")) ls.scope = parse_scope(sc->string, path);
    if (const Json *th = item.find("threshold")) ls.threshold = (float)th->number;
    if (const Json *sf = item.find("safe"); sf && sf->kind == Json::Kind::Bool && sf->boolean)
      ls.flags |= ENGINE_PROBE_FLAG_SAFE;
    if (const Json *cs = item.find("scale")) ls.cal_scale = (float)cs->number;
    if (const Json *cb = item.find("bias")) ls.cal_bias = (float)cb->number;
    ls.is_linear = true;
    ls.linear_index = base_linear_idx + (uint32_t)i;
    out.push_back(std::move(ls));
  }
}

void apply_calibration(const std::string &dir, ProbeWeights &pw, uint32_t model_layers, uint32_t model_hidden) {
  const std::string cal_path = dir + "/calibration.json";
  if (::access(cal_path.c_str(), R_OK) != 0) return;
  const Json cal = parse_json(read_text_file(cal_path));
  if (cal.kind != Json::Kind::Object) throw std::runtime_error(cal_path + ": root must be an object");

  if (const Json *mn = cal.find("model_name"); mn && mn->kind == Json::Kind::String && !mn->string.empty())
    pw.spec.model_name = mn->string;
  if (const Json *rw = cal.find("roll_window"); rw && rw->kind == Json::Kind::Number) {
    const uint32_t v = (uint32_t)rw->number;
    if (v == 0 || v > 256) throw std::runtime_error(cal_path + ": roll_window must be in 1..256");
    pw.spec.roll_window = v;
  }

  // Optional sidecar directory of extra linear/drift alarms sharing the base probe's tapped layers.
  if (const Json *ex = cal.find("extra_alarms"); ex && ex->kind == Json::Kind::String && !ex->string.empty()) {
    const std::string ex_dir = ex->string[0] == '/' ? ex->string : dir + "/" + ex->string;
    ProbeWeights extra = load_probe(ex_dir, model_layers, model_hidden);
    if (extra.spec.kind != ProbeKind::Linear)
      throw std::runtime_error(cal_path + ": extra_alarms must be a c0ven_probe_v1 checkpoint");
    if (extra.spec.layer_ids != pw.spec.layer_ids || extra.spec.hidden_size != pw.spec.hidden_size)
      throw std::runtime_error(cal_path + ": extra_alarms must match the base probe's layer_ids and hidden_size");
    if (pw.spec.labels.size() + extra.spec.labels.size() > ENGINE_MAX_PROBE_LABELS)
      throw std::runtime_error(cal_path + ": combined labels exceed ENGINE_MAX_PROBE_LABELS");
    const uint32_t off = pw.spec.linear_labels;
    pw.linear_w.insert(pw.linear_w.end(), extra.linear_w.begin(), extra.linear_w.end());
    pw.linear_b.insert(pw.linear_b.end(), extra.linear_b.begin(), extra.linear_b.end());
    for (ProbeLabelSpec ls : extra.spec.labels) {
      ls.linear_index += off;
      pw.spec.labels.push_back(std::move(ls));
    }
    pw.spec.linear_labels += extra.spec.linear_labels;
  }

  auto find_label = [&](const std::string &name) -> ProbeLabelSpec * {
    for (auto &ls : pw.spec.labels)
      if (ls.name == name) return &ls;
    return nullptr;
  };
  if (const Json *ths = cal.find("thresholds"); ths && ths->kind == Json::Kind::Object) {
    for (const auto &[name, v] : ths->object)
      if (auto *ls = find_label(name); ls && v.kind == Json::Kind::Number) ls->threshold = (float)v.number;
  }
  if (const Json *temps = cal.find("temperature"); temps && temps->kind == Json::Kind::Object) {
    for (const auto &[name, v] : temps->object)
      if (auto *ls = find_label(name); ls && v.kind == Json::Kind::Number && v.number > 0.0)
        ls->cal_scale = (float)(1.0 / v.number);
  }
  if (const Json *sc = cal.find("scale"); sc && sc->kind == Json::Kind::Object) {
    for (const auto &[name, v] : sc->object)
      if (auto *ls = find_label(name); ls && v.kind == Json::Kind::Number) ls->cal_scale = (float)v.number;
  }
  if (const Json *bs = cal.find("bias"); bs && bs->kind == Json::Kind::Object) {
    for (const auto &[name, v] : bs->object)
      if (auto *ls = find_label(name); ls && v.kind == Json::Kind::Number) ls->cal_bias = (float)v.number;
  }
}

void matvec_bf16_cpu(const uint16_t *W, const uint16_t *x, uint32_t rows, uint32_t cols, float *out) {
  for (uint32_t r = 0; r < rows; r++) {
    const uint16_t *wr = W + (size_t)r * cols;
    float s = 0.0f;
    for (uint32_t k = 0; k < cols; k++) s += bf16_to_f32(wr[k]) * bf16_to_f32(x[k]);
    out[r] = s;
  }
}

}  // namespace

engine_probe_info ProbeSpec::to_c() const {
  engine_probe_info out{};
  copy_name(out.model_name, model_name);
  copy_name(out.kind, kind_name);
  out.n_labels = std::min<uint32_t>((uint32_t)labels.size(), ENGINE_MAX_PROBE_LABELS);
  out.window = window;
  out.roll_window = roll_window;
  for (uint32_t i = 0; i < out.n_labels; i++) {
    copy_name(out.labels[i].name, labels[i].name);
    out.labels[i].threshold = labels[i].threshold;
    out.labels[i].scope = labels[i].scope;
    out.labels[i].flags = labels[i].flags;
  }
  return out;
}

ProbeWeights load_probe(const std::string &dir, uint32_t model_layers, uint32_t model_hidden) {
  const std::string cfg_path = dir + "/config.json";
  const Json cfg = parse_json(read_text_file(cfg_path));
  if (cfg.kind != Json::Kind::Object) throw std::runtime_error(cfg_path + ": root must be an object");

  ProbeWeights pw;
  ProbeSpec &sp = pw.spec;
  sp.kind_name = cfg.at("model_type").string;
  sp.hidden_size = (uint32_t)cfg.at("hidden_size").number;
  if (sp.hidden_size == 0) throw std::runtime_error(cfg_path + ": hidden_size must be positive");
  if (model_hidden > 0 && sp.hidden_size != model_hidden)
    throw std::runtime_error(cfg_path + ": hidden_size " + std::to_string(sp.hidden_size) +
                             " != model hidden_size " + std::to_string(model_hidden));
  sp.layer_ids = read_layer_ids(cfg, cfg_path, model_layers);
  if (const Json *mn = cfg.find("model_name"); mn && mn->kind == Json::Kind::String && !mn->string.empty())
    sp.model_name = mn->string;
  if (const Json *rw = cfg.find("roll_window"); rw && rw->kind == Json::Kind::Number) {
    sp.roll_window = (uint32_t)rw->number;
    if (sp.roll_window == 0 || sp.roll_window > 256)
      throw std::runtime_error(cfg_path + ": roll_window must be in 1..256");
  }
  if (const Json *rn = cfg.find("rms_norm_taps"); rn && rn->kind == Json::Kind::Bool) sp.rms_norm_taps = rn->boolean;
  if (const Json *eps = cfg.find("rms_norm_eps"); eps && eps->kind == Json::Kind::Number && eps->number > 0)
    sp.norm_eps = (float)eps->number;

  const SafeFile sf(dir + "/model.safetensors");
  const uint64_t feat = sp.feat_dim();

  if (sp.kind_name == "sing_probe_attn") {
    sp.kind = ProbeKind::SingProbeAttn;
    if (sp.model_name.empty()) sp.model_name = "singprobe";
    sp.num_heads = (uint32_t)cfg.at("num_attention_heads").number;
    sp.head_dim = (uint32_t)cfg.at("head_dim").number;
    sp.window = (uint32_t)cfg.at("sliding_window").number;
    sp.base_labels = (uint32_t)cfg.at("num_labels").number;
    if (sp.num_heads == 0 || sp.head_dim == 0 || sp.window == 0 || sp.base_labels == 0 ||
        sp.base_labels > ENGINE_MAX_PROBE_LABELS)
      throw std::runtime_error(cfg_path + ": invalid sing_probe_attn dimensions");
    const uint64_t qd = sp.q_dim(), kvd = sp.kv_dim();
    pw.proj_q = sf.read_bf16("proj_q.weight", {qd, feat});
    pw.proj_k = sf.read_bf16("proj_k.weight", {kvd, feat});
    pw.proj_v = sf.read_bf16("proj_v.weight", {kvd, feat});
    pw.o_proj = sf.read_bf16("o_proj.weight", {qd, qd});
    pw.norm = sf.read_f32("norm.weight", {qd});
    pw.cls_w = sf.read_f32("classifier.weight", {sp.base_labels, qd});
    pw.cls_b = sf.has("classifier.bias") ? sf.read_f32("classifier.bias", {sp.base_labels})
                                         : std::vector<float>(sp.base_labels, 0.0f);
    sp.labels = default_singprobe_labels(sp.base_labels);
  } else if (sp.kind_name == "sing_probe_mlp") {
    sp.kind = ProbeKind::SingProbeMlp;
    if (sp.model_name.empty()) sp.model_name = "singprobe-mlp";
    sp.base_labels = (uint32_t)cfg.at("num_labels").number;
    if (const Json *inter = cfg.find("intermediate_size"); inter && inter->kind == Json::Kind::Number)
      sp.intermediate_size = (uint32_t)inter->number;
    else if (const Json *hd = cfg.find("mlp_hidden_size"); hd && hd->kind == Json::Kind::Number)
      sp.intermediate_size = (uint32_t)hd->number;
    if (sp.intermediate_size == 0 || sp.base_labels == 0 || sp.base_labels > ENGINE_MAX_PROBE_LABELS)
      throw std::runtime_error(cfg_path + ": invalid sing_probe_mlp dimensions");
    pw.fc1_w = sf.read_bf16("fc1.weight", {sp.intermediate_size, feat});
    pw.fc1_b = sf.has("fc1.bias") ? sf.read_f32("fc1.bias", {sp.intermediate_size})
                                  : std::vector<float>(sp.intermediate_size, 0.0f);
    pw.fc2_w = sf.read_f32("fc2.weight", {sp.base_labels, sp.intermediate_size});
    pw.fc2_b = sf.has("fc2.bias") ? sf.read_f32("fc2.bias", {sp.base_labels})
                                  : std::vector<float>(sp.base_labels, 0.0f);
    sp.labels = default_singprobe_labels(sp.base_labels);
  } else if (sp.kind_name == "c0ven_probe_v1") {
    sp.kind = ProbeKind::Linear;
    if (sp.model_name.empty()) sp.model_name = "c0ven-probe";
    parse_c0ven_labels(cfg, cfg_path, sp.labels, 0);
    if (sp.labels.size() > ENGINE_MAX_PROBE_LABELS)
      throw std::runtime_error(cfg_path + ": too many probe labels");
    sp.linear_labels = (uint32_t)sp.labels.size();
    pw.linear_w = sf.read_bf16("linear.weight", {sp.linear_labels, feat});
    pw.linear_b = sf.has("linear.bias") ? sf.read_f32("linear.bias", {sp.linear_labels})
                                        : std::vector<float>(sp.linear_labels, 0.0f);
    for (uint32_t i = 0; i < sp.linear_labels; i++) sp.labels[i].linear_bias = pw.linear_b[i];
  } else {
    throw std::runtime_error(cfg_path + ": unsupported probe model_type '" + sp.kind_name + "'");
  }

  // Optional label-name overrides directly in config.json (e.g. id2label or labels).
  if (sp.kind != ProbeKind::Linear) {
    if (const Json *i2l = cfg.find("id2label"); i2l && i2l->kind == Json::Kind::Object) {
      for (uint32_t i = 0; i < sp.base_labels; i++)
        if (const Json *v = i2l->find(std::to_string(i)); v && v->kind == Json::Kind::String && !v->string.empty())
          sp.labels[i].name = v->string;
    }
  }

  apply_calibration(dir, pw, model_layers, model_hidden);
  return pw;
}

void probe_rms_norm_bf16(const float *in, uint32_t len, float eps, uint16_t *out) {
  float ss = 0.0f;
  for (uint32_t i = 0; i < len; i++) ss += in[i] * in[i];
  const float inv = 1.0f / std::sqrt(ss / (float)len + eps);
  for (uint32_t i = 0; i < len; i++) out[i] = f32_to_bf16(in[i] * inv);
}

void ProbeCpuSlot::init(const ProbeSpec &spec) {
  start = 0;
  if (spec.kind == ProbeKind::SingProbeAttn) {
    k_ring.assign((size_t)spec.window * spec.kv_dim(), 0);
    v_ring.assign((size_t)spec.window * spec.kv_dim(), 0);
  } else {
    k_ring.clear();
    v_ring.clear();
  }
}

void ProbeCpuSlot::reset(uint32_t pos) {
  start = pos;
  std::fill(k_ring.begin(), k_ring.end(), 0);
  std::fill(v_ring.begin(), v_ring.end(), 0);
}

void probe_eval_cpu(const ProbeWeights &pw, ProbeCpuSlot &slot, uint32_t pos, const uint16_t *feat_bf16, float *out) {
  const ProbeSpec &sp = pw.spec;
  const uint32_t feat = sp.feat_dim();

  if (sp.kind == ProbeKind::SingProbeAttn) {
    const uint32_t H = sp.num_heads, D = sp.head_dim, qd = sp.q_dim(), W = sp.window;
    std::vector<float> q(qd), k(D), v(D);
    matvec_bf16_cpu(pw.proj_q.data(), feat_bf16, qd, feat, q.data());
    matvec_bf16_cpu(pw.proj_k.data(), feat_bf16, D, feat, k.data());
    matvec_bf16_cpu(pw.proj_v.data(), feat_bf16, D, feat, v.data());

    const uint32_t cell = pos % W;
    uint16_t *kc = slot.k_ring.data() + (size_t)cell * D;
    uint16_t *vc = slot.v_ring.data() + (size_t)cell * D;
    for (uint32_t d = 0; d < D; d++) {
      kc[d] = f32_to_bf16(k[d]);
      vc[d] = f32_to_bf16(v[d]);
    }

    const uint32_t lo = std::max(slot.start, pos + 1 > W ? pos + 1 - W : 0u);
    const uint32_t nctx = pos >= lo ? pos - lo + 1 : 0;
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> scores(nctx);
    std::vector<float> attn(qd, 0.0f);
    for (uint32_t h = 0; h < H; h++) {
      const float *qh = q.data() + (size_t)h * D;
      float mx = -INFINITY;
      for (uint32_t i = 0; i < nctx; i++) {
        const uint16_t *ki = slot.k_ring.data() + (size_t)((lo + i) % W) * D;
        float dot = 0.0f;
        for (uint32_t d = 0; d < D; d++) dot += qh[d] * bf16_to_f32(ki[d]);
        scores[i] = dot * scale;
        mx = std::max(mx, scores[i]);
      }
      float sum = 0.0f;
      for (uint32_t i = 0; i < nctx; i++) {
        scores[i] = std::exp(scores[i] - mx);
        sum += scores[i];
      }
      const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
      float *ah = attn.data() + (size_t)h * D;
      for (uint32_t i = 0; i < nctx; i++) {
        const float w = scores[i] * inv;
        const uint16_t *vi = slot.v_ring.data() + (size_t)((lo + i) % W) * D;
        for (uint32_t d = 0; d < D; d++) ah[d] += w * bf16_to_f32(vi[d]);
      }
    }

    // Residual projection + RMSNorm (no +1) + linear classifier.
    std::vector<float> hid(qd, 0.0f);
    for (uint32_t r = 0; r < qd; r++) {
      const uint16_t *orow = pw.o_proj.data() + (size_t)r * qd;
      float s = q[r];
      for (uint32_t c = 0; c < qd; c++) s += bf16_to_f32(orow[c]) * attn[c];
      hid[r] = s;
    }
    float ss = 0.0f;
    for (uint32_t r = 0; r < qd; r++) ss += hid[r] * hid[r];
    const float rinv = 1.0f / std::sqrt(ss / (float)qd + sp.norm_eps);
    for (uint32_t r = 0; r < qd; r++) hid[r] = hid[r] * rinv * pw.norm[r];
    for (uint32_t l = 0; l < sp.base_labels; l++) {
      const float *cw = pw.cls_w.data() + (size_t)l * qd;
      float s = pw.cls_b[l];
      for (uint32_t r = 0; r < qd; r++) s += cw[r] * hid[r];
      out[l] = s;
    }
  } else if (sp.kind == ProbeKind::SingProbeMlp) {
    const uint32_t mid = sp.intermediate_size;
    std::vector<float> h(mid);
    matvec_bf16_cpu(pw.fc1_w.data(), feat_bf16, mid, feat, h.data());
    for (uint32_t i = 0; i < mid; i++) h[i] = silu_f(h[i] + pw.fc1_b[i]);
    for (uint32_t l = 0; l < sp.base_labels; l++) {
      const float *w2 = pw.fc2_w.data() + (size_t)l * mid;
      float s = pw.fc2_b[l];
      for (uint32_t i = 0; i < mid; i++) s += w2[i] * h[i];
      out[l] = s;
    }
  }

  // Linear and drift alarms (for drift, emit raw dot product without bias; tracker subtracts before/after + bias).
  if (sp.linear_labels > 0) {
    std::vector<float> dots(sp.linear_labels);
    matvec_bf16_cpu(pw.linear_w.data(), feat_bf16, sp.linear_labels, feat, dots.data());
    for (uint32_t i = 0; i < sp.n_labels(); i++) {
      const ProbeLabelSpec &ls = sp.labels[i];
      if (!ls.is_linear) continue;
      const uint32_t li = ls.linear_index;
      out[i] = ls.scope == ENGINE_PROBE_SCOPE_DRIFT ? dots[li] : dots[li] + pw.linear_b[li];
    }
  }
}

void ProbeTracker::init(const ProbeSpec *spec, uint32_t prompt_len, uint32_t flags, const engine_token_span *spans,
                        uint32_t n_spans) {
  reset();
  spec_ = spec;
  prompt_len_ = prompt_len;
  flags_ = flags;
  if (!active()) return;
  const uint32_t nl = spec_->n_labels();
  roll_window_ = std::max(1u, spec_->roll_window);
  in_ring_.assign((size_t)roll_window_ * nl, 0.0f);
  out_ring_.assign((size_t)roll_window_ * nl, 0.0f);
  in_sum_.assign(nl, 0.0);
  out_sum_.assign(nl, 0.0);
  in_max_.assign(nl, 0.0f);
  out_max_.assign(nl, 0.0f);
  last_dots_.assign(nl, 0.0f);
  spans_.clear();
  for (uint32_t i = 0; i < n_spans && i < ENGINE_MAX_PROBE_SPANS; i++) {
    if (spans[i].start >= spans[i].end || spans[i].start >= prompt_len) continue;
    SpanState st;
    st.start = spans[i].start;
    st.end = std::min(spans[i].end, prompt_len);
    st.pre_dots.assign(nl, 0.0f);
    if (st.start == 0) st.pre_ready = true;  // baseline 0 when untrusted span starts at token 0
    spans_.push_back(std::move(st));
  }
}

void ProbeTracker::reset() {
  spec_ = nullptr;
  prompt_len_ = 0;
  flags_ = 0;
  roll_window_ = 10;
  in_count_ = 0;
  out_count_ = 0;
  in_ring_.clear();
  out_ring_.clear();
  in_sum_.clear();
  out_sum_.clear();
  in_max_.clear();
  out_max_.clear();
  in_flagged_ = false;
  out_flagged_ = false;
  spans_.clear();
  last_dots_.clear();
  has_last_dots_ = false;
}

void ProbeTracker::update_drift(uint32_t pos, const float *raw) {
  const uint32_t nl = spec_->n_labels();
  for (SpanState &st : spans_) {
    if (st.start > 0 && pos == st.start - 1) {
      for (uint32_t i = 0; i < nl; i++)
        if (spec_->labels[i].scope == ENGINE_PROBE_SCOPE_DRIFT) st.pre_dots[i] = raw[i];
      st.pre_ready = true;
    }
    if (pos == st.end - 1 && st.pre_ready && !st.post_ready) {
      st.post_ready = true;
      for (uint32_t i = 0; i < nl; i++) {
        const ProbeLabelSpec &ls = spec_->labels[i];
        if (ls.scope != ENGINE_PROBE_SCOPE_DRIFT) continue;
        const float diff = raw[i] - st.pre_dots[i] + ls.linear_bias;
        const float p = sigmoid_f(ls.cal_scale * diff + ls.cal_bias);
        if (p > in_max_[i]) in_max_[i] = p;
        if (!(ls.flags & ENGINE_PROBE_FLAG_SAFE) && in_max_[i] >= ls.threshold) in_flagged_ = true;
      }
    }
  }
}

void ProbeTracker::feed_token(uint32_t pos, const float *raw, uint32_t n) {
  if (!active()) return;
  const uint32_t nl = spec_->n_labels();
  if (n < nl) return;

  for (uint32_t i = 0; i < nl; i++)
    if (spec_->labels[i].scope == ENGINE_PROBE_SCOPE_DRIFT) last_dots_[i] = raw[i];
  has_last_dots_ = true;

  const bool is_input = pos < prompt_len_;
  if (is_input) {
    update_drift(pos, raw);
    const uint32_t idx = in_count_ % roll_window_;
    const bool full = in_count_ >= roll_window_;
    const uint32_t denom = full ? roll_window_ : (in_count_ + 1);
    const bool recompute = (in_count_ & 4095u) == 4095u;
    for (uint32_t i = 0; i < nl; i++) {
      const ProbeLabelSpec &ls = spec_->labels[i];
      if (ls.scope != ENGINE_PROBE_SCOPE_INPUT && ls.scope != ENGINE_PROBE_SCOPE_BOTH) continue;
      const float p = sigmoid_f(ls.cal_scale * raw[i] + ls.cal_bias);
      const size_t slot_off = (size_t)idx * nl + i;
      if (full) in_sum_[i] -= in_ring_[slot_off];
      in_ring_[slot_off] = p;
      in_sum_[i] += p;
      if (recompute) {
        double s = 0.0;
        for (uint32_t k = 0; k < denom; k++) s += in_ring_[(size_t)k * nl + i];
        in_sum_[i] = s;
      }
      const float mean = (float)(in_sum_[i] / (double)denom);
      if (mean > in_max_[i]) in_max_[i] = mean;
      if (!(ls.flags & ENGINE_PROBE_FLAG_SAFE) && in_max_[i] >= ls.threshold) in_flagged_ = true;
    }
    in_count_++;
  } else {
    const uint32_t idx = out_count_ % roll_window_;
    const bool full = out_count_ >= roll_window_;
    const uint32_t denom = full ? roll_window_ : (out_count_ + 1);
    const bool recompute = (out_count_ & 4095u) == 4095u;
    for (uint32_t i = 0; i < nl; i++) {
      const ProbeLabelSpec &ls = spec_->labels[i];
      if (ls.scope != ENGINE_PROBE_SCOPE_OUTPUT && ls.scope != ENGINE_PROBE_SCOPE_BOTH) continue;
      const float p = sigmoid_f(ls.cal_scale * raw[i] + ls.cal_bias);
      const size_t slot_off = (size_t)idx * nl + i;
      if (full) out_sum_[i] -= out_ring_[slot_off];
      out_ring_[slot_off] = p;
      out_sum_[i] += p;
      if (recompute) {
        double s = 0.0;
        for (uint32_t k = 0; k < denom; k++) s += out_ring_[(size_t)k * nl + i];
        out_sum_[i] = s;
      }
      const float mean = (float)(out_sum_[i] / (double)denom);
      if (mean > out_max_[i]) out_max_[i] = mean;
      if (!(ls.flags & ENGINE_PROBE_FLAG_SAFE) && out_max_[i] >= ls.threshold) out_flagged_ = true;
    }
    out_count_++;
  }
}

ProbeTracker::Snapshot ProbeTracker::snapshot() const {
  Snapshot s;
  if (!active()) return s;
  s.n_labels = spec_->n_labels();
  s.roll_window = roll_window_;
  s.in_count = in_count_;
  s.in_ring = in_ring_;
  s.in_sum = in_sum_;
  s.in_max = in_max_;
  s.in_flagged = in_flagged_;
  s.last_dots = last_dots_;
  s.has_last_dots = has_last_dots_;
  s.spans = spans_;
  return s;
}

void ProbeTracker::restore(const Snapshot &snap, uint32_t checkpoint_pos) {
  if (!active() || snap.n_labels != spec_->n_labels() || snap.roll_window != roll_window_) return;
  in_count_ = snap.in_count;
  in_ring_ = snap.in_ring;
  in_sum_ = snap.in_sum;
  in_max_ = snap.in_max;
  in_flagged_ = snap.in_flagged;
  last_dots_ = snap.last_dots;
  has_last_dots_ = snap.has_last_dots;
  for (SpanState &st : spans_) {
    if (st.end <= checkpoint_pos) {
      st.post_ready = true;
      continue;
    }
    if (has_last_dots_ && checkpoint_pos > 0 && st.start == checkpoint_pos) {
      st.pre_dots = last_dots_;
      st.pre_ready = true;
      continue;
    }
    for (const SpanState &prev : snap.spans) {
      if (prev.start == st.start && prev.pre_ready) {
        st.pre_dots = prev.pre_dots;
        st.pre_ready = true;
        break;
      }
    }
  }
}

engine_slot_probe ProbeTracker::to_c() const {
  engine_slot_probe out{};
  if (!active()) return out;
  const uint32_t nl = std::min<uint32_t>(spec_->n_labels(), ENGINE_MAX_PROBE_LABELS);
  out.n_labels = nl;
  out.input_flagged = in_flagged_ ? 1u : 0u;
  out.output_flagged = out_flagged_ ? 1u : 0u;
  for (uint32_t i = 0; i < nl; i++) {
    out.input_scores[i] = in_max_[i];
    out.output_scores[i] = out_max_[i];
  }
  return out;
}

}  // namespace engine
