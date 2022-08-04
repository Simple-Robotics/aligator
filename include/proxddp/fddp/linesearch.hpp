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
  static void run(F phi, M model, Scalar phi0, VerboseLevel verbose,
                  LinesearchParams<Scalar> &ls_params, Scalar &alpha_opt) {
    static Scalar th_accept_step_ = 0.1;
    static Scalar th_accept_neg_step_ = 2.0;
    Scalar atry = 1.;

    // backtrack until going under alpha_min
    do {
      Scalar dVreal = phi(atry) - phi0;
      Scalar dVmodel = model(atry) - phi0;
      // check if descending
      if (dVmodel <= 0.) {
        if (dVreal <= th_accept_step_ * dVmodel)
          break;
      } else {
        // accept a small increase in cost
        if (dVreal >= th_accept_neg_step_ * dVmodel)
          break;
      }
      atry *= ls_params.ls_beta;
    } while (atry >= ls_params.alpha_min);
    alpha_opt = std::max(ls_params.alpha_min, atry);
  }
};

} // namespace proxddp
