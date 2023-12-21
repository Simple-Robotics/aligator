#pragma once

#include "proxddp/context.hpp"

namespace aligator {

extern template struct ExplicitDynamicsModelTpl<context::Scalar>;

extern template struct ExplicitDynamicsDataTpl<context::Scalar>;

} // namespace aligator
