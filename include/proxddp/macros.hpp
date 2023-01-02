#pragma once

#ifdef PROXDDP_EIGEN_CHECK_MALLOC
#define PROXDDP_EIGEN_ALLOW_MALLOC(allowed)                                    \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#else
#define PROXDDP_EIGEN_ALLOW_MALLOC(allowed)
#endif

/// @brief Entering performance-critical code.
#define PROXDDP_NOMALLOC_BEGIN PROXDDP_EIGEN_ALLOW_MALLOC(false)
/// @brief Exiting performance-critical code.
#define PROXDDP_NOMALLOC_END PROXDDP_EIGEN_ALLOW_MALLOC(true)

#define PROXDDP_INLINE inline __attribute__((always_inline))
