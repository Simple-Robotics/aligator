#pragma once

#include <omp.h>

namespace aligator {

/// Utilities to set parallelism options.
namespace omp {

inline auto get_available_threads() {
  return (std::size_t)omp_get_max_threads();
}

/// Get the current number of threads.
inline auto get_current_threads() { return (std::size_t)omp_get_num_threads(); }

/// Get the current thread ID.
inline auto get_thread_id() { return (std::size_t)omp_get_thread_num(); }

inline void set_default_options(std::size_t num_threads, int dynamic = 1) {
  omp_set_num_threads((int)num_threads);
  omp_set_dynamic(dynamic);
}

} // namespace omp

} // namespace aligator
