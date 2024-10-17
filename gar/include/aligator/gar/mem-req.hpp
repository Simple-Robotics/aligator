#pragma once

#include <cstdlib>
#include <vector>
#include <cassert>

namespace aligator::gar {
typedef unsigned int uint;

/// \brief Utility class for asking for sets of bytes aligned to a given value.
/// It allows preallocation of a block of memory. It is not like a memory pool.
struct MemReq {
  MemReq(uint alignment) noexcept;

  MemReq &addBytes(uint size) noexcept;

  /// \brief Request for bytes from an array.
  template <typename T, typename... Args>
  MemReq &addArray(uint n1, Args... args) noexcept {
    uint size = n1;
    std::initializer_list<uint> dims{static_cast<uint>(args)...};
    for (auto s : dims) {
      size *= s;
    }
    addBytes(size * sizeof(T));
    return *this;
  }

  void *allocate() const;

  /// Advance the pointer to the memory buffer to the next chunk.
  template <typename T> void advance(T *&memory) {
    assert(_cursor != _chunkSizes.size() && "Reached end of the chunks.");
    memory += _chunkSizes[_cursor++] / sizeof(T);
  }

  /// Reset the cursor for the advance() method.
  void reset() { _cursor = 0; }

  uint totalBytes() const { return _totalBytes; }

private:
  uint _align;
  uint _totalBytes{0};
  std::vector<uint> _chunkSizes;
  uint _cursor{0};
};

} // namespace aligator::gar
