#pragma one

#include "proxddp/fwd.hpp"

namespace proxddp {

template <typename Scalar> struct ResultsFDDPTpl : ResultsBaseTpl<Scalar> {

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ResultsBaseTpl<Scalar>;
  using BlockXs = Eigen::Block<MatrixXs, -1, -1>;

  using Base::gains_;
  using Base::us_;
  using Base::xs_;

  decltype(auto) getFeedforward(std::size_t i) { return gains_[i].col(0); }
  decltype(auto) getFeedforward(std::size_t i) const {
    return gains_[i].col(0);
  }

  decltype(auto) getFeedback(std::size_t i) {
    const long ndx = this->gains_[i].cols() - 1;
    return gains_[i].rightCols(ndx);
  }

  decltype(auto) getFeedback(std::size_t i) const {
    const long ndx = this->gains_[i].cols() - 1;
    return gains_[i].rightCols(ndx);
  }

  explicit ResultsFDDPTpl(const TrajOptProblemTpl<Scalar> &problem);
};

template <typename Scalar>
ResultsFDDPTpl<Scalar>::ResultsFDDPTpl(
    const TrajOptProblemTpl<Scalar> &problem) {
  using StageModel = StageModelTpl<Scalar>;

  const std::size_t nsteps = problem.numSteps();
  xs_.resize(nsteps + 1);
  us_.resize(nsteps);

  xs_default_init(problem, xs_);
  us_default_init(problem, us_);

  gains_.resize(nsteps);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];

    const int ndx = sm.ndx1();
    const int nu = sm.nu();
    const int ndual = sm.numDual();

    gains_[i] = MatrixXs::Zero(nu, ndx + 1);
  }
}

} // namespace proxddp
