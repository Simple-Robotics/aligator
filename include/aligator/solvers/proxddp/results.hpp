/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/solvers/results-base.hpp"
#include <fmt/ostream.h>
#include <sstream>

namespace aligator {

/// @brief    Results holder struct.
template <typename _Scalar> struct ResultsTpl final : ResultsBaseTpl<_Scalar> {
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

  explicit ResultsTpl()
      : Base() {}

  ResultsTpl(const ResultsTpl &) = default;
  ResultsTpl &operator=(const ResultsTpl &) = default;

  ResultsTpl(ResultsTpl &&) = default;
  ResultsTpl &operator=(ResultsTpl &&) = default;

  /// @brief    Create the results struct from a problem (TrajOptProblemTpl)
  /// instance.
  explicit ResultsTpl(const TrajOptProblemTpl<Scalar> &problem);

  void cycleAppend(const TrajOptProblemTpl<Scalar> &problem,
                   const ConstVectorRef &x0);
};

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const ResultsTpl<Scalar> &self) {
  return oss << fmt::format("{}", self);
}

} // namespace aligator

template <typename Scalar> struct fmt::formatter<aligator::ResultsTpl<Scalar>> {
  constexpr auto parse(format_parse_context &ctx) const
      -> decltype(ctx.begin()) {
    return ctx.end();
  }

  auto format(const aligator::ResultsTpl<Scalar> &self,
              format_context &ctx) const -> decltype(ctx.out()) {
    auto s = self.printBase();
    return fmt::format_to(ctx.out(), "Results {{{}\n}}", s);
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./results.txx"
#endif
