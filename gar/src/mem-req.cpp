#include "aligator/gar/mem-req.hpp"
#include <stdlib.h>

namespace aligator::gar {
MemReq::MemReq(uint alignment) noexcept : _align(alignment) {}

MemReq &MemReq::addBytes(uint size) noexcept {
  uint memsize = (size + _align - 1) / _align * _align;
  _totalBytes += memsize;
  _chunkSizes.push_back(memsize);
  return *this;
}

void *MemReq::allocate() const {
  return std::aligned_alloc(_align, _totalBytes);
}
} // namespace aligator::gar
