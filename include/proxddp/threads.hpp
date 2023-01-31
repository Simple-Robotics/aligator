#pragma once

#include <omp.h>

namespace proxddp {

inline int getNumAvailableThreads() { return omp_get_max_threads(); }
inline int getCurrentThreads() { return omp_get_num_threads(); }

} // namespace proxddp
