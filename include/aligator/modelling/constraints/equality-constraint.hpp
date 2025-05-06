/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/constraint-set.hpp"

namespace aligator {

///
/// @brief   Equality constraints \f$c(x) = 0\f$.
///
/// @details This class implements the set associated with equality
/// constraints\f$ c(x) = 0 \f$, where \f$c : \calX \to \RR^p\f$ is a residual
/// function.
///
template <typename _Scalar>
struct EqualityConstraintTpl : ConstraintSetTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  EqualityConstraintTpl() = default;
  EqualityConstraintTpl(const EqualityConstraintTpl &) = default;
  EqualityConstraintTpl &operator=(const EqualityConstraintTpl &) = default;
  EqualityConstraintTpl(EqualityConstraintTpl &&) = default;
  EqualityConstraintTpl &operator=(EqualityConstraintTpl &&) = default;

  using Base = ConstraintSetTpl<Scalar>;
  using ActiveType = typename Base::ActiveType;

  bool disableGaussNewton() const { return true; }

  inline void projection(const ConstVectorRef &, VectorRef zout) const {
    zout.setZero();
  }

  inline void normalConeProjection(const ConstVectorRef &z,
                                   VectorRef zout) const {
    zout = z;
  }

  inline void applyProjectionJacobian(const ConstVectorRef &,
                                      MatrixRef Jout) const {
    Jout.setZero();
  }

  inline void applyNormalConeProjectionJacobian(const ConstVectorRef &,
                                                MatrixRef) const {
    return; // do nothing
  }

  inline void computeActiveSet(const ConstVectorRef &,
                               Eigen::Ref<ActiveType> out) const {
    out.array() = true;
  }
};

} // namespace aligator
