/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/results-base.hpp"

namespace proxddp {

/// @brief    Results holder struct.
template <typename _Scalar> struct ResultsTpl : ResultsBaseTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ResultsBaseTpl<Scalar>;
  using Base::conv;
  using Base::gains_;
  using Base::lams;
  using Base::num_iters;
  using Base::us;
  using Base::xs;

  std::size_t al_iter = 0;

  /// @brief    Create the results struct from a problem (TrajOptProblemTpl)
  /// instance.
  explicit ResultsTpl(const TrajOptProblemTpl<Scalar> &problem);
};

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const ResultsTpl<Scalar> &self) {
  oss << "Results {";
  self.printBase(oss);
  return oss << fmt::format("\n  al_iters:     {:d},", self.al_iter) << "\n}";
}

} // namespace proxddp

#include "proxddp/core/results.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/results.txx"
#endif
