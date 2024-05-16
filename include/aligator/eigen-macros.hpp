/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "macros.hpp"

#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
/// @warning Unless using versions of Eigen past 3.4.x, NOT SUPPORTED IN
/// MULTITHREADED CONTEXTS
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)                                   \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
/// @brief Set nomalloc for the scope. Previous malloc status will be restored
/// upon exiting the scope.
/// @warning Unless using versions of Eigen past 3.4.x, NOT SUPPORTED IN
/// MULTITHREADED CONTEXTS
#define ALIGATOR_NOMALLOC_SCOPED                                               \
  const ::aligator::internal::scoped_nomalloc ___aligator_nomalloc_zone {}
/// @brief Restore the previous nomalloc status.
#define ALIGATOR_NOMALLOC_RESTORE                                              \
  ALIGATOR_EIGEN_ALLOW_MALLOC(::aligator::internal::get_cached_malloc_status())
#else
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)
#define ALIGATOR_NOMALLOC_SCOPED
#define ALIGATOR_NOMALLOC_RESTORE
#endif

/// @brief Entering performance-critical code.
#define ALIGATOR_NOMALLOC_BEGIN ALIGATOR_EIGEN_ALLOW_MALLOC(false)
/// @brief Exiting performance-critical code.
#define ALIGATOR_NOMALLOC_END ALIGATOR_EIGEN_ALLOW_MALLOC(true)

#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
namespace aligator::internal {
thread_local static bool g_cached_malloc_status = true;

inline void set_malloc_status(bool status) { g_cached_malloc_status = status; }

inline void save_malloc_status() {
  set_malloc_status(
#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
      ::Eigen::internal::is_malloc_allowed()
#else
      false
#endif
  );
}

inline bool get_cached_malloc_status() { return g_cached_malloc_status; }

struct scoped_nomalloc {
  scoped_nomalloc(const scoped_nomalloc &) = delete;
  scoped_nomalloc(scoped_nomalloc &&) = delete;
  ALIGATOR_INLINE scoped_nomalloc() {
    save_malloc_status();
    ALIGATOR_EIGEN_ALLOW_MALLOC(false);
  }
  // reset to value from before the scope
  ~scoped_nomalloc() { ALIGATOR_NOMALLOC_RESTORE; }
};

} // namespace aligator::internal
#endif
