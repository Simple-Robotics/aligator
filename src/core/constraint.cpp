#include "proxddp/core/constraint.hpp"

namespace proxddp {

template struct StageConstraintTpl<context::Scalar>;

template struct ConstraintStackTpl<context::Scalar>;

template struct ConstraintALWeightStrategy<context::Scalar>;

} // namespace proxddp
