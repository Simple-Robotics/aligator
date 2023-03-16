#pragma once

#include "proxddp/context.hpp"

namespace proxddp {

extern template struct StateErrorResidualTpl<context::Scalar>;
extern template struct ControlErrorResidualTpl<context::Scalar>;

} // namespace proxddp
