#pragma once

#include "aligator/core/results-base.hpp"

namespace aligator {

template <typename Scalar> struct ResultsFDDPTpl : ResultsBaseTpl<Scalar> {

  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ResultsBaseTpl<Scalar>;
  using BlockXs = Eigen::Block<MatrixXs, -1, -1>;

  using Base::gains_;
  using Base::us;
  using Base::xs;

  ResultsFDDPTpl() : Base() {}
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

    gains_[i].setZero(nu, ndx + 1);
  }
  this->m_isInitialized = true;
}

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./results.txx"
#endif
