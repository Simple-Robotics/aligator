/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"

namespace aligator {

template <typename Scalar> struct TrajectoryTpl {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  std::vector<VectorXs> xs;
  std::vector<VectorXs> us;
  std::vector<VectorXs> vs;
  std::vector<VectorXs> lbdas;
  long horizon() const { return xs.size() - 1; }
};

} // namespace aligator
