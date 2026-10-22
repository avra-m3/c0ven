// Each work-item writes its global id plus a bias: proves a module built by ocloc loads and runs on a card.
__kernel void smoke(__global uint *out, uint bias) {
  size_t i = get_global_id(0);
  out[i] = (uint)i + bias;
}
