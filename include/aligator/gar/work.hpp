/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include <cassert>

namespace aligator {
namespace gar {
using uint = unsigned int;

struct workrange_t {
  uint beg;
  uint end;
};

/// @brief Get a balanced work range corresponding to a horizon @p horz, thread
/// ID @p tid, and number of threads @p num_threads.
constexpr workrange_t get_work(uint horz, uint thread_id, uint num_threads) {
  uint start = thread_id * (horz + 1) / num_threads;
  uint stop = (thread_id + 1) * (horz + 1) / num_threads;
  assert(stop <= horz + 1);
  return {start, stop};
}

} // namespace gar
} // namespace aligator
