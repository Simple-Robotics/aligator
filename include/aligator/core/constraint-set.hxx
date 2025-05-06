/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/constraint-set.hpp"

namespace aligator {

template <typename Scalar>
void ConstraintSetTpl<Scalar>::applyProjectionJacobian(const ConstVectorRef &z,
                                                       MatrixRef Jout) const {
  const int nr = (int)z.size();
  assert(nr == Jout.rows());
  ActiveType active_set(nr);
  computeActiveSet(z, active_set);
  for (int i = 0; i < nr; i++) {
    /// active constraints -> projector onto the constraint set is zero
    if (active_set(i)) {
      Jout.row(i).setZero();
    }
  }
}

template <typename Scalar>
void ConstraintSetTpl<Scalar>::applyNormalConeProjectionJacobian(
    const ConstVectorRef &z, MatrixRef Jout) const {
  const int nr = (int)z.size();
  assert(nr == Jout.rows());
  ActiveType active_set(nr);
  computeActiveSet(z, active_set);
  for (int i = 0; i < nr; i++) {
    /// inactive constraint -> normal cone projection is zero
    if (!active_set(i)) {
      Jout.row(i).setZero();
    }
  }
}

} // namespace aligator
