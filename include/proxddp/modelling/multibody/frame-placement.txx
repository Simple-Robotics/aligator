#pragma once

#include "./frame-placement.hpp"
#include "proxddp/context.hpp"

namespace proxddp {

extern template struct FramePlacementResidualTpl<context::Scalar>;
extern template struct FramePlacementDataTpl<context::Scalar>;

} // namespace proxddp
