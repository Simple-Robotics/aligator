#pragma once

#include "proxddp/fddp/results.hpp"

namespace proxddp {

template <typename Scalar>
ResultsFDDPTpl<Scalar>::ResultsFDDPTpl(
    const TrajOptProblemTpl<Scalar> &problem) {
  using StageModel = StageModelTpl<Scalar>;

  const std::size_t nsteps = problem.numSteps();
  xs.resize(nsteps + 1);
  us.resize(nsteps);

  xs_default_init(problem, xs);
  us_default_init(problem, us);

  gains_.resize(nsteps);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];

    const int ndx = sm.ndx1();
    const int nu = sm.nu();

    gains_[i] = MatrixXs::Zero(nu, ndx + 1);
  }
  this->m_isInitialized = true;
}

} // namespace proxddp