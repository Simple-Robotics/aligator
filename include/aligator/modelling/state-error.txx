#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/state-error.hpp"

namespace aligator {

extern template struct StateErrorResidualTpl<context::Scalar>;
extern template struct ControlErrorResidualTpl<context::Scalar>;

} // namespace aligator
