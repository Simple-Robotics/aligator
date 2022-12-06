/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/linesearch.hpp"

namespace proxddp {

/// @brief    The backtracking linesearch from FDDP (Mastalli et al).
/// @details  The conditions that are checked for are not exactly the Goldstein
/// conditions.
/// @return an std::pair
template <typename Scalar, typename F, typename M>
PROX_INLINE std::pair<Scalar, Scalar>
fddp_goldstein_linesearch(F &&phi, M &&model, const Scalar phi0,
                          const typename Linesearch<Scalar>::Options &ls_params,
                          Scalar th_grad, Scalar &d1) {
  Scalar th_accept_step_ = 0.1;
  Scalar th_accept_neg_step_ = 2.0;
  const Scalar beta = ls_params.contraction_min;
  Scalar atry = 1.;
  Scalar phitry = phi0;

  // backtrack until going under alpha_min
  do {
    try {
      phitry = phi(atry);
    } catch (const ::proxddp::RuntimeError &) {
      atry *= beta;
      continue;
    }
    Scalar dVreal = phitry - phi0;
    Scalar dVmodel = model(atry) - phi0;
    // check if descent direction
    bool descent_ok = (dVmodel <= 0.);
    descent_ok &= (-d1 < th_grad) || (dVreal <= th_accept_step_ * dVmodel);
    // or accept small increase in cost;
    bool ascent_ok = dVreal <= th_accept_neg_step_ * dVmodel;
    if (descent_ok || ascent_ok) {
      break;
    }
    atry *= beta;
  } while (atry >= ls_params.alpha_min);
  return {atry, phitry};
}

} // namespace proxddp
