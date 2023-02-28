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

  ResultsFDDPTpl() : Base() {}
  explicit ResultsFDDPTpl(const TrajOptProblemTpl<Scalar> &problem);
};

} // namespace proxddp

#include "proxddp/fddp/results.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/fddp/results.txx"
#endif
