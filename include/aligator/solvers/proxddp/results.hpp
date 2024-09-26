/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/results-base.hpp"
#include <fmt/ostream.h>

namespace aligator {

/// @brief    Results holder struct.
template <typename _Scalar> struct ResultsTpl : ResultsBaseTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ResultsBaseTpl<Scalar>;
  using Base::conv;
  using Base::gains_;
  using Base::num_iters;
  using Base::us;
  using Base::xs;

  /// Problem co-states
  std::vector<VectorXs> lams;
  /// Path constraint multipliers
  std::vector<VectorXs> vs;
  /// Proximal/AL iteration count
  std::size_t al_iter = 0;

  ResultsTpl() : Base() {}

  ResultsTpl(const ResultsTpl &) = delete;
  ResultsTpl &operator=(const ResultsTpl &) = delete;

  ResultsTpl(ResultsTpl &&) = default;
  ResultsTpl &operator=(ResultsTpl &&) = default;

  /// @brief    Create the results struct from a problem (TrajOptProblemTpl)
  /// instance.
  explicit ResultsTpl(const TrajOptProblemTpl<Scalar> &problem);

  void cycleAppend(const TrajOptProblemTpl<Scalar> &problem,
                   const Eigen::VectorXd &x0);
};

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const ResultsTpl<Scalar> &self) {
  oss << "Results {";
  self.printBase(oss);
  return oss << fmt::format("\n  al_iters:     {:d},", self.al_iter) << "\n}";
}

} // namespace aligator

template <typename Scalar>
struct fmt::formatter<aligator::ResultsTpl<Scalar>> : fmt::ostream_formatter {};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./results.txx"
#endif
