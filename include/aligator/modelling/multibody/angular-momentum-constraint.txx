#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/angular-momentum-constraint.hpp"

namespace aligator {

extern template struct AngularMomentumConstraintResidualTpl<context::Scalar>;
extern template struct AngularMomentumConstraintDataTpl<context::Scalar>;

} // namespace aligator
