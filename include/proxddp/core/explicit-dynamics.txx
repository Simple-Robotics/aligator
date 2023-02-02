#pragma once

#include "proxddp/context.hpp"


namespace proxddp {

extern template struct ExplicitDynamicsModelTpl<context::Scalar>;

extern template struct ExplicitDynamicsDataTpl<context::Scalar>;

} // namespace proxddp
