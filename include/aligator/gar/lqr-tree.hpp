#pragma once

#include "./util.hpp"
#include <array>
#include <cmath>

namespace aligator {
using std::size_t;

struct LQRTree {
  using size2 = std::array<size_t, 2>;

  LQRTree(size_t length) : horz_(length) { buildTree(); }

  /// Build the tree structure, allocate the data...
  /// TODO: point the data to somewhere else
  void buildTree() { maxDepth_ = intLog2(horz_); }

  size_t maxDepth() const { return maxDepth_; }

  size_t getIndex(size_t depth, size_t i) const {
    size_t s = intExp2(depth);
    return s - 1 + i;
  }

  size_t getIndexDepth(size_t index) const { return intLog2(index); }

  size_t getLeafIndex(size_t i) const { return getIndex(maxDepth_ - 1, i); }

  size2 getChildren(size_t index) const {
    return {2 * index + 1, 2 * index + 2};
  }

  size_t getIndexParent(size_t index) const {
    if (index == 0)
      return 0;
    size_t half = (index - 1) / 2;
    return half;
  }

  size_t getLevelIndex(size_t index) {
    size_t depth = getIndexDepth(index);
    size_t s = intExp2(depth);
    return index - s;
  }

private:
  size_t horz_;
  size_t maxDepth_;
};

} // namespace aligator
