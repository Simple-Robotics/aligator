#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/traj-opt-problem.hpp"

namespace proxddp {

namespace {
  using StateErrorResidual = StateErrorResidualTpl<context::Scalar>;
}

extern template
TrajOptProblemTpl<context::Scalar>::TrajOptProblemTpl(const context::VectorXs&, const std::vector<shared_ptr<context::StageModel>> &, const shared_ptr<context::CostBase> &);

extern template
TrajOptProblemTpl<context::Scalar>::TrajOptProblemTpl(const context::VectorXs&, const int, const shared_ptr<context::Manifold> &, const shared_ptr<context::CostBase> &);

extern template
TrajOptProblemTpl<context::Scalar>::TrajOptProblemTpl(const StateErrorResidual &, const int, const shared_ptr<context::CostBase> &);

} // namespace proxddp
