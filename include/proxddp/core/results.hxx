#pragma once

#include "proxddp/core/solver-util.hpp"

#include <fmt/format.h>

namespace proxddp {

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss,
                         const ResultsBaseTpl<Scalar> &self) {
  oss << "Results {" << fmt::format("\n  num_iters:    {:d},", self.num_iters)
      << fmt::format("\n  converged:    {},", self.conv)
      << fmt::format("\n  traj. cost:   {:.3e},", self.traj_cost_)
      << fmt::format("\n  merit.value:  {:.3e},", self.merit_value_)
      << fmt::format("\n  prim_infeas:  {:.3e},", self.prim_infeas)
      << fmt::format("\n  dual_infeas:  {:.3e},", self.dual_infeas);
  oss << "\n}";
  return oss;
}

template <typename Scalar>
ResultsTpl<Scalar>::ResultsTpl(const TrajOptProblemTpl<Scalar> &problem) {

  const std::size_t nsteps = problem.numSteps();
  gains_.reserve(nsteps);
  xs_default_init(problem, xs);
  us_default_init(problem, us);
  lams.reserve(nsteps + 1);
  {
    const int ndual = problem.init_state_error.nr;
    lams.push_back(VectorXs::Zero(ndual));
  }
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &stage = *problem.stages_[i];
    const int nprim = stage.numPrimal();
    const int ndual = stage.numDual();
    gains_.push_back(MatrixXs::Zero(nprim + ndual, stage.ndx1() + 1));
    lams.push_back(VectorXs::Zero(ndual));
  }

  if (problem.term_constraint_) {
    const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
    const int ndx = tc.func->ndx1;
    const int ndual = tc.func->nr;
    lams.push_back(VectorXs::Zero(ndual));
    gains_.push_back(MatrixXs::Zero(ndual, ndx + 1));
  }
  assert(xs.size() == nsteps + 1);
  assert(us.size() == nsteps);
}
} // namespace proxddp
