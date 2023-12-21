#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/state-error.hpp"

namespace aligator {

extern template struct StateErrorResidualTpl<context::Scalar>;
extern template struct ControlErrorResidualTpl<context::Scalar>;

} // namespace aligator
