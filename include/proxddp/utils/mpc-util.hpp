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
template <typename T, typename Alloc>
void rotate_vec_left(std::vector<T, Alloc> &v,
                     typename std::vector<T, Alloc>::iterator end,
                     long n_head = 0) {
  auto beg = std::next(v.begin(), n_head);
  std::rotate(beg, beg + 1, end);
}

/// @overload rotate_vec_left
/// This overload supposes we want to rotate the std::vector until its
/// termination.
template <typename T, typename Alloc>
void rotate_vec_left(std::vector<T, Alloc> &v, long n_head = 0) {
  rotate_vec_left(v, v.end(), n_head);
}

} // namespace proxddp
