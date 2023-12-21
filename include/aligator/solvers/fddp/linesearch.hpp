/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/linesearch.hpp"

namespace aligator {

/// @brief    The backtracking linesearch from FDDP (Mastalli et al).
/// @details  The conditions that are checked for are not exactly the Goldstein
/// conditions.
/// @return an std::pair
template <typename Scalar, typename F, typename M>
ALIGATOR_INLINE std::pair<Scalar, Scalar> fddp_goldstein_linesearch(
    F &&phi, M &&model, const Scalar phi0,
    const typename Linesearch<Scalar>::Options &ls_params, Scalar th_grad,
    Scalar &d1, Scalar th_accept_step = 0.1, Scalar th_accept_neg_step = 2.0) {
  const Scalar beta = ls_params.contraction_min;
  Scalar atry = 1.;
  Scalar phitry = phi0;
  Scalar dVreal, dVmodel;
  constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

  // backtrack until going under alpha_min
  while (true) {
    if (atry < ls_params.alpha_min + eps) {
      break;
    }
    try {
      phitry = phi(atry);
    } catch (const ::aligator::RuntimeError &) {
      atry *= beta;
      continue;
    }
    dVreal = phitry - phi0;
    dVmodel = model(atry) - phi0;
    // check if descent direction
    if (dVmodel < 0.) {
      if (std::abs(d1) < th_grad || dVreal <= th_accept_step * dVmodel) {
        break;
      }
    } else {
      // or accept small increase in cost;
      if (dVreal <= th_accept_neg_step * dVmodel) {
        break;
      }
    }
    atry *= beta;
    atry = std::max(atry, ls_params.alpha_min);
  }
  return {atry, phitry};
}

} // namespace aligator
