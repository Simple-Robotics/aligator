/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA

#include "aligator/solvers/proxddp/merit-function.hxx"

namespace aligator {

template struct ALFunction<context::Scalar>;
template context::Scalar costDirectionalDerivative<context::Scalar>(
    const WorkspaceTpl<context::Scalar> &workspace,
    const TrajOptDataTpl<context::Scalar> &prob_data);

} // namespace aligator
