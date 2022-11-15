/// @file
/// @brief Utils for model-predictive control.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <vector>
#include <algorithm>

namespace proxddp {

/// @brief Simply rotate an entire std::vector to the left.
/// @tparam T
/// @tparam Alloc
/// @param  n_head The length of the vector (at the head) to keep.
/// @param  n_tail The length of the vector (at the end) to keep.
template <typename T, typename Alloc>
void rotate_vec_left(std::vector<T, Alloc> &v, long n_head = 0,
                     long n_tail = 0) {
  if (v.size() > 0) {
    std::rotate(v.begin() + n_head, v.begin() + n_head + 1, v.end() - n_tail);
  }
};

} // namespace proxddp
