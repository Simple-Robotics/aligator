/// @file
/// @brief Utils for model-predictive control.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <vector>
#include <algorithm>

namespace proxddp {

/// @brief Simply rotate an entire std::vector to the left.
/// @tparam T
/// @tparam Alloc
/// @param  n_head The length of the vector (at the head) to keep.
/// @param  n_tail The length of the vector (at the tail ) to keep.
template <typename T, typename Alloc>
void rotate_vec_left(std::vector<T, Alloc> &v, long n_head = 0,
                     long n_tail = 0) {
  auto beg = std::next(v.begin(), n_head);
  auto end = std::prev(v.end(), n_tail);
  std::rotate(beg, beg + 1, end);
}

} // namespace proxddp
