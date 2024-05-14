/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "macros.hpp"

#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)                                   \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#define ALIGATOR_NOMALLOC_SCOPED                                               \
  const ::aligator::internal::scoped_nomalloc ___aligator_nomalloc_zone {}
#else
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)
#define ALIGATOR_NOMALLOC_SCOPED
#endif

/// @brief Entering performance-critical code.
#define ALIGATOR_NOMALLOC_BEGIN ALIGATOR_EIGEN_ALLOW_MALLOC(false)
/// @brief Exiting performance-critical code.
#define ALIGATOR_NOMALLOC_END ALIGATOR_EIGEN_ALLOW_MALLOC(true)

namespace aligator::internal {
static struct {
  bool value;
} g_malloc_status;

inline bool set_malloc_status(bool status) {
  g_malloc_status.value = status;
  return g_malloc_status.value;
}

inline void save_malloc_status() {
  set_malloc_status(
#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
      ::Eigen::internal::is_malloc_allowed()
#else
      false
#endif
  );
}

inline bool get_malloc_status() { return g_malloc_status.value; }

struct scoped_nomalloc {
  scoped_nomalloc(const scoped_nomalloc &) = delete;
  scoped_nomalloc(scoped_nomalloc &&) = delete;
  ALIGATOR_INLINE scoped_nomalloc() {
    save_malloc_status();
    ALIGATOR_EIGEN_ALLOW_MALLOC(true);
  }
  // reset to value from before the scope
  ~scoped_nomalloc() {
    ALIGATOR_MAYBE_UNUSED bool active = get_malloc_status();
    ALIGATOR_EIGEN_ALLOW_MALLOC(active);
  }
};

} // namespace aligator::internal