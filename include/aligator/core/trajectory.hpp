#pragma once

#include "aligator/math.hpp"

namespace aligator {

template <typename Scalar> struct TrajectoryTpl {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  std::vector<VectorXs> xs;
  std::vector<VectorXs> us;
  std::vector<VectorXs> vs;
  std::vector<VectorXs> lbdas;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/context.hpp"

namespace aligator {
extern template struct TrajectoryTpl<context::Scalar>;
}
#endif
