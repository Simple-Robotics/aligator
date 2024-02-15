#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/kinodynamics-wrapper.hpp"

namespace aligator {

extern template struct KinodynamicsWrapperResidualTpl<context::Scalar>;
extern template struct KinodynamicsWrapperDataTpl<context::Scalar>;

} // namespace aligator
