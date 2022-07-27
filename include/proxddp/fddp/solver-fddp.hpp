#pragma once

#include "proxddp/core/solver-base.hpp"

namespace proxddp {

template <typename _Scalar> struct SolverFDDP {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = TrajOptProblemTpl<Scalar>;

  bool run(const Problem &problem,
           const std::vector<VectorXs> &xs_init = DEFAULT_VECTOR<Scalar>,
           const std::vector<VectorXs> &us_init = DEFAULT_VECTOR<Scalar>) {
    return false;
  }
};

} // namespace proxddp

#include "proxddp/fddp/solver-fddp.hxx"
