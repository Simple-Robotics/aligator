#pragma once

#include "proxddp/core/solver-base.hpp"

#include <fmt/core.h>

namespace proxddp {

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss,
                         const ResultsBaseTpl<Scalar> &self) {
  oss << "Results {";
  oss << fmt::format("\n  numiters   :  {:d},", self.num_iters);
  oss << fmt::format("\n  converged  :  {},", self.conv);
  oss << fmt::format("\n  traj. cost :  {:.3e},", self.traj_cost_)
      << fmt::format("\n  merit.value:  {:.3e},", self.merit_value_)
      << fmt::format("\n  prim_infeas:  {:.3e},", self.primal_infeasibility)
      << fmt::format("\n  dual_infeas:  {:.3e},", self.dual_infeasibility);
  oss << "\n}";
  return oss;
}

template <typename Scalar>
ResultsTpl<Scalar>::ResultsTpl(const TrajOptProblemTpl<Scalar> &problem) {

  const std::size_t nsteps = problem.numSteps();
  gains_.reserve(nsteps);
  xs_default_init(problem, xs_);
  us_default_init(problem, us_);
  lams_.reserve(nsteps + 1);
  co_state_.reserve(nsteps);
  const int ndual = problem.init_state_error.nr;
  lams_.push_back(VectorXs::Ones(ndual));
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &stage = *problem.stages_[i];
    const int nprim = stage.numPrimal();
    const int ndual = stage.numDual();
    gains_.push_back(MatrixXs::Zero(nprim + ndual, stage.ndx1() + 1));
    lams_.push_back(VectorXs::Ones(ndual));
    const int nr = stage.ndx2();
    co_state_.push_back(lams_[i + 1].head(nr));
  }

  if (problem.term_constraint_) {
    const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
    const int ndx = tc.func_->ndx1;
    const int ndual = tc.func_->nr;
    lams_.push_back(VectorXs::Zero(ndual));
    gains_.push_back(MatrixXs::Zero(ndual, ndx + 1));
  }
  assert(xs_.size() == nsteps + 1);
  assert(us_.size() == nsteps);
}
} // namespace proxddp
