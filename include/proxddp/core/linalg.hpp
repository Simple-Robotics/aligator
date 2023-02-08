/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/math.hpp"

namespace proxddp {

template <typename Scalar> struct iterative_refinement_impl {

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  template <typename DecompoAlgo, typename OutputType>
  static bool run(const DecompoAlgo &ldlt, const MatrixXs &mat,
                  const MatrixXs &rhs, MatrixXs &err, OutputType &Xout,
                  const Scalar refinement_threshold,
                  const std::size_t max_refinement_steps) {

    std::size_t it = 0;

    Xout = -rhs;
    ldlt.solveInPlace(Xout);

    err = -rhs;
    err.noalias() -= mat * Xout;

    while (math::infty_norm(err) > refinement_threshold) {

      if (it >= max_refinement_steps) {
        return false;
      }

      ldlt.solveInPlace(err);
      Xout += err;

      // update residual
      err = -rhs;
      err.noalias() -= mat * Xout;

      it++;
    }
    return true;
  }
};

} // namespace proxddp