#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/traj-opt-problem.hpp"

#include <ostream>

namespace proxddp {

template <typename _Scalar> struct ResultsBaseTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  std::size_t num_iters = 0;
  bool conv = false;

  Scalar traj_cost_ = 0.;
  Scalar merit_value_ = 0.;
  /// Overall primal infeasibility/constraint violation for the
  /// TrajOptProblemTpl.
  Scalar primal_infeasibility = 0.;
  /// Overall dual infeasibility measure for the TrajOptProblemTpl.
  Scalar dual_infeasibility = 0.;

  /// Riccati gains
  std::vector<MatrixXs> gains_;
  /// States
  std::vector<VectorXs> xs_;
  /// Controls
  std::vector<VectorXs> us_;
  /// Problem Lagrange multipliers
  std::vector<VectorXs> lams_;
  /// Dynamics' co-states
  std::vector<VectorRef> co_state_;

  int get_ndx1() const { return this->gains_[i].cols() - 1; }

  decltype(auto) getFeedforward(std::size_t i) {
    return this->gains_[i].col(0);
  }

  decltype(auto) getFeedforward(std::size_t i) const {
    return this->gains_[i].col(0);
  }

  decltype(auto) getFeedback(std::size_t i) {
    return this->gains_[i].rightCols(this->get_ndx1(i));
  }

  decltype(auto) getFeedback(std::size_t i) const {
    return this->gains_[i].rightCols(this->get_ndx1(i));
  }
};

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const ResultsBaseTpl<Scalar> &self);

/// @brief    Results holder struct.
template <typename _Scalar> struct ResultsTpl : ResultsBaseTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ResultsBaseTpl<Scalar>;
  using Base::co_state_;
  using Base::conv;
  using Base::gains_;
  using Base::lams_;
  using Base::num_iters;
  using Base::us_;
  using Base::xs_;

  std::size_t al_iter = 0;

  /// @brief    Create the results struct from a problem (TrajOptProblemTpl)
  /// instance.
  explicit ResultsTpl(const TrajOptProblemTpl<Scalar> &problem);
};

} // namespace proxddp

#include "proxddp/core/solver-results.hxx"
