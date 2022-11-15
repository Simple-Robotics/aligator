#pragma once

#ifdef PROXDDP_EIGEN_CHECK_MALLOC
#define PROXDDP_EIGEN_ALLOW_MALLOC(allowed)                                    \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#else
#define PROXDDP_EIGEN_ALLOW_MALLOC(allowed)
#endif
