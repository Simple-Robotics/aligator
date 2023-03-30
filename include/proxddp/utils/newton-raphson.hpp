#pragma once

#include "proxddp/fwd.hpp"

#include <Eigen/LU>

namespace proxddp {

/// @brief  Newton-Raphson procedure, e.g. to compute forward dynamics from
/// implicit functions.
template <typename Scalar> struct NewtonRaphson {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Manifold = ManifoldAbstractTpl<Scalar>;

  struct DataView {
    VectorRef f0;  // fun value
    VectorRef dx0; // workspace for the newton step
    MatrixRef J0;  // fun jacobian
  };

  template <typename Fun, typename JacFun>
  static bool run(const Manifold &space, Fun &&fun, JacFun &&jac_fun,
                  const ConstVectorRef &xinit, VectorRef xout, DataView &data,
                  Scalar eps = 1e-6, std::size_t max_iters = 1000,
                  VerboseLevel = VerboseLevel::QUIET) {
    const Scalar alpha_min = 1e-4;
    const Scalar ls_beta = 0.8;
    const Scalar ar_c1 = 1e-2;

    xout = xinit;
    VectorRef &f0 = data.f0;
    VectorRef &dx = data.dx0;
    MatrixRef &Jf0 = data.J0;
    fun(xout, f0);

    Scalar err = f0.norm();
    bool conv = false;
    for (std::size_t i = 0; i < max_iters; i++) {
      if (err <= eps) {
        conv = true;
        break;
      }
      dx = Jf0.lu().solve(-f0);

      Scalar alpha = 1.;
      while (alpha > alpha_min) {
        space.integrate(xout, alpha * dx, xout);
        fun(xout, f0);
        err = f0.norm();
        if (err <= (1. - ar_c1) * err) {
          break;
        }
        alpha *= ls_beta;
      }

      jac_fun(xout, Jf0);
    }
    return conv;
  }
};

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/utils/newton-raphson.txx"
#endif
