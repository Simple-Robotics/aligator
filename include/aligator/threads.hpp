#pragma once

#ifdef ALIGATOR_MULTITHREADING
#include <omp.h>
#endif

namespace aligator {

/// Utilities to set parallelism options.
namespace omp {

#ifdef ALIGATOR_MULTITHREADING
inline std::size_t get_available_threads() {
  return std::size_t(omp_get_max_threads());
}

/// Get the current number of threads.
inline std::size_t get_current_threads() {
  return std::size_t(omp_get_num_threads());
}

/// Get the current thread ID.
inline std::size_t get_thread_id() { return std::size_t(omp_get_thread_num()); }

inline void set_default_options(std::size_t num_threads, bool dynamic = true) {
  omp_set_num_threads(int(num_threads));
  omp_set_dynamic(int(dynamic));
}
#else
inline std::size_t get_available_threads() { return 0; }

/// Get the current number of threads.
inline std::size_t get_current_threads() { return 0; }

/// Get the current thread ID.
inline std::size_t get_thread_id() { return 0; }

inline void set_default_options(std::size_t, bool = true) {}

#endif

} // namespace omp

} // namespace aligator
