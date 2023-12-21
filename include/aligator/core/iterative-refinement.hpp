/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"

namespace aligator {

template <typename Scalar> struct IterativeRefinementVisitor {

  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  template <typename DecompoAlgo> bool operator()(const DecompoAlgo &ldlt) {

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

  const MatrixXs &mat;
  const MatrixXs &rhs;
  MatrixXs &err;
  Eigen::Ref<MatrixXs> Xout;
  const Scalar refinement_threshold;
  const std::size_t max_refinement_steps;
};

} // namespace aligator
