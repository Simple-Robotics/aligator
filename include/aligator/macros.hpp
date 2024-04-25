#pragma once

#if __cplusplus >= 201703L
#define ALIGATOR_WITH_CPP_17
#endif

#if __cplusplus >= 201402L
#define ALIGATOR_WITH_CPP_14
#endif

#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)                                   \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#define ALIGATOR_NOMALLOC_SCOPED
#else
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)
#define ALIGATOR_NOMALLOC_SCOPED                                               \
  ::aligator::internal::scoped_nomalloc {}
#endif

/// @brief Entering performance-critical code.
#define ALIGATOR_NOMALLOC_BEGIN ALIGATOR_EIGEN_ALLOW_MALLOC(false)
/// @brief Exiting performance-critical code.
#define ALIGATOR_NOMALLOC_END ALIGATOR_EIGEN_ALLOW_MALLOC(true)

#define ALIGATOR_INLINE inline __attribute__((always_inline))

#ifdef ALIGATOR_WITH_CPP_17
#define ALIGATOR_MAYBE_UNUSED [[maybe_unused]]
#elif defined(_MSC_VER) && !defined(__clang__)
#define ALIGATOR_MAYBE_UNUSED
#else
#define ALIGATOR_MAYBE_UNUSED __attribute__((__unused__))
#endif

namespace aligator {
namespace internal {
struct scoped_nomalloc {
  scoped_nomalloc() { ALIGATOR_NOMALLOC_BEGIN; }
  ~scoped_nomalloc() { ALIGATOR_NOMALLOC_END; }
};
} // namespace internal
} // namespace aligator
