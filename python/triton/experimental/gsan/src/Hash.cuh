#include <stdint.h>

namespace gsan {
namespace {

uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

uint32_t value_hash32(uint32_t x) {
  x *= 0xcc9e2d51u;
  x = rotl32(x, 15);
  x *= 0x1b873593u;
  return x;
}

uint32_t hash_finalize32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

uint32_t hash_combine32(uint32_t h, uint32_t v) {
  h ^= value_hash32(v);
  h = rotl32(h, 13);
  h = h * 5u + 0xe6546b64u;
  return h;
}

uint32_t hash2x32(uint32_t a, uint32_t b, uint32_t seed) {
  uint32_t h = seed;
  h = hash_combine32(h, a);
  h = hash_combine32(h, b);
  h ^= 8u; // length in bytes (2 * 4)
  return hash_finalize32(h);
}

} // namespace
} // namespace gsan
