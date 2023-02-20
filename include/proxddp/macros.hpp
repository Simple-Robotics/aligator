#pragma once

#if __cplusplus >= 201703L
#define PROXDDP_WITH_CPP_17
#endif

#if __cplusplus >= 201402L
#define PROXDDP_WITH_CPP_14
#endif

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

#ifdef PROXDDP_WITH_CPP_17
#define PROXDDP_MAYBE_UNUSED [[maybe_unused]]
#elif defined(_MSC_VER) && !defined(__clang__)
#define PROXDDP_MAYBE_UNUSED
#else
#define PROXDDP_MAYBE_UNUSED __attribute__((__unused__))
#endif
