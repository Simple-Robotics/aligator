#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/contact-force.hpp"

namespace aligator {

extern template struct ContactForceResidualTpl<context::Scalar>;
extern template struct ContactForceDataTpl<context::Scalar>;

} // namespace aligator
