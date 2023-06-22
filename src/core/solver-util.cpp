#include "proxddp/core/solver-util.hpp"
#include "proxddp/core/workspace.hpp"

namespace proxddp {
template void
xs_default_init<context::Scalar>(const context::TrajOptProblem &,
                                 std::vector<context::VectorXs> &);

template void
us_default_init<context::Scalar>(const context::TrajOptProblem &,
                                 std::vector<context::VectorXs> &);

template void check_trajectory_and_assign<context::Scalar>(
    const context::TrajOptProblem &, const std::vector<context::VectorXs> &,
    const std::vector<context::VectorXs> &, std::vector<context::VectorXs> &,
    std::vector<context::VectorXs> &);

template void computeLagrangianDerivatives<context::Scalar>(
    const context::TrajOptProblem &problem, context::Workspace &workspace,
    const context::VectorOfVectors &lams);

} // namespace proxddp
