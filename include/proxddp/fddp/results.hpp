#pragma once

#include "proxddp/core/results-base.hpp"

namespace proxddp {

template <typename Scalar> struct ResultsFDDPTpl : ResultsBaseTpl<Scalar> {

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ResultsBaseTpl<Scalar>;
  using BlockXs = Eigen::Block<MatrixXs, -1, -1>;

  using Base::gains_;
  using Base::us;
  using Base::xs;

  explicit ResultsFDDPTpl(const TrajOptProblemTpl<Scalar> &problem);
};

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
}

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/fddp/results.txx"
#endif
