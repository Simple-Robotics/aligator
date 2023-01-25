#include "proxddp/core/traj-opt-problem.hpp"

namespace proxddp {

template TrajOptProblemTpl<context::Scalar>::TrajOptProblemTpl(
    const context::VectorXs &,
    const std::vector<shared_ptr<context::StageModel>> &,
    const shared_ptr<context::CostBase> &);

template TrajOptProblemTpl<context::Scalar>::TrajOptProblemTpl(
    const context::VectorXs &, const int, const shared_ptr<context::Manifold> &,
    const shared_ptr<context::CostBase> &);

template TrajOptProblemTpl<context::Scalar>::TrajOptProblemTpl(
    const StateErrorResidual &, const int,
    const shared_ptr<context::CostBase> &);

} // namespace proxddp
