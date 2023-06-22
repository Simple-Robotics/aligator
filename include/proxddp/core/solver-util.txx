#pragma once

#include "proxddp/context.hpp"
#include "./solver-util.hpp"

namespace proxddp {

extern template void
xs_default_init<context::Scalar>(const context::TrajOptProblem &,
                                 std::vector<context::VectorXs> &);

extern template void
us_default_init<context::Scalar>(const context::TrajOptProblem &,
                                 std::vector<context::VectorXs> &);

extern template void check_trajectory_and_assign<context::Scalar>(
    const context::TrajOptProblem &, const std::vector<context::VectorXs> &,
    const std::vector<context::VectorXs> &, std::vector<context::VectorXs> &,
    std::vector<context::VectorXs> &);

extern template void computeLagrangianDerivatives<context::Scalar>(const context::TrajOptProblem &problem, context::Workspace &workspace, const context::VectorOfVectors &lams);

} // namespace proxddp
