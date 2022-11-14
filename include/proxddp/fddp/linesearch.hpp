#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/linesearch.hpp"

namespace proxddp {

/// @brief    The backtracking linesearch from FDDP (Mastalli et al).
/// @details  The conditions that are checked for are not exactly the Goldstein
/// conditions.
template <typename Scalar> struct FDDPGoldsteinLinesearch {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  template <typename F, typename M>
  static Scalar run(F phi, M model, Scalar phi0,
                    typename Linesearch<Scalar>::Options &ls_params, Scalar d1,
                    Scalar th_grad) {
    static Scalar th_accept_step_ = 0.1;
    static Scalar th_accept_neg_step_ = 2.0;
    Scalar beta = ls_params.contraction_min;
    Scalar atry = 1.;

    // backtrack until going under alpha_min
    do {
      Scalar dVreal = 0.;
      try {
        dVreal = phi(atry) - phi0;
      } catch (const std::runtime_error &) {
        atry *= beta;
        continue;
      }
      Scalar dVmodel = model(atry) - phi0;
      // check if descent direction
      if (dVmodel <= 0.) {
        if (std::abs(d1) < th_grad || dVreal <= th_accept_step_ * dVmodel)
          break;
      } else {
        // accept a small increase in cost
        if (dVreal >= th_accept_neg_step_ * dVmodel)
          break;
      }
      atry *= beta;
    } while (atry >= ls_params.alpha_min);
    atry = std::max(ls_params.alpha_min, atry);
    return atry;
  }
};

} // namespace proxddp
