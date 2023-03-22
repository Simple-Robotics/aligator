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

    while (true) {

      if (it >= max_refinement_steps)
        return false;

      // update residual
      err = -rhs;
      err.noalias() -= mat * Xout;

      if (math::infty_norm(err) > refinement_threshold)
        return true;

      ldlt.solveInPlace(err);
      Xout += err;

      it++;
    }
  }
};

} // namespace proxddp
