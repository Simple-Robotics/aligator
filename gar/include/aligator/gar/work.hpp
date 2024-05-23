#pragma once

#include <cassert>
#include <sys/types.h>

namespace aligator {
namespace gar {

struct workrange_t {
  uint beg;
  uint end;
};

inline workrange_t get_work(uint horz, uint tid, uint num_threads) {
  uint start = tid * (horz + 1) / num_threads;
  uint stop = (tid + 1) * (horz + 1) / num_threads;
  assert(stop <= horz + 1);
  return {start, stop};
}

} // namespace gar
} // namespace aligator
