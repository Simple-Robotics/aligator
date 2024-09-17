#pragma once

#include "aligator/context.hpp"
#include "merit-function.hpp"

namespace aligator {

extern template struct PDALFunction<context::Scalar>;
extern template context::Scalar costDirectionalDerivative<context::Scalar>(
    const WorkspaceTpl<context::Scalar> &workspace,
    const TrajOptDataTpl<context::Scalar> &prob_data);

} // namespace aligator
