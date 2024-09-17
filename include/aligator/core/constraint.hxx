/// @file constraint.hxx
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/constraint.hpp"

namespace aligator {

template <typename Scalar> void ConstraintStackTpl<Scalar>::clear() {
  funcs.clear();
  sets.clear();
  indices_ = {0};
  dims_.clear();
  total_dim_ = 0;
}

} // namespace aligator
