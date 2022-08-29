#pragma once

#include "proxddp/fwd.hpp"

#include <Eigen/LU>

namespace proxddp {

/// @brief  Newton-Raphson procedure, e.g. to compute forward dynamics from
/// implicit functions.
template <typename Scalar> struct NewtonRaphson {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using VectorRef = typename math_types<Scalar>::VectorRef;
  using ConstVectorRef = typename math_types<Scalar>::ConstVectorRef;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  template <typename Fun, typename JacFun>
  static bool run(const Manifold &man, Fun &fun, JacFun &jac_fun,
                  const ConstVectorRef xinit, VectorRef xout,
                  const Scalar eps = 1e-6, const std::size_t MAXITERS = 1000,
                  VerboseLevel verbose = VerboseLevel::QUIET) {
    xout = xinit;
    VectorXs f0 = fun(xout);
    VectorXs dx(f0);
    MatrixXs Jf0 = jac_fun(xout);
    Scalar error = f0.norm();
    bool conv = false;
    const Scalar alpha_min = 1e-4;
    const Scalar ls_beta = 0.8;
    const Scalar ar_c1 = 1e-2;
    for (std::size_t i = 0; i < MAXITERS; i++) {
      if (error <= eps) {
        conv = true;
        break;
      }
      dx = Jf0.lu().solve(-f0);

      Scalar alpha = 1.;
      while (alpha > alpha_min) {
        man.integrate(xout, alpha * dx, xout);
        f0 = fun(xout);
        Scalar new_error = f0.norm();
        if (new_error <= (1. - ar_c1) * error) {
          break;
        }
        alpha *= ls_beta;
      }

      error = f0.norm();
      Jf0 = jac_fun(xout);
    }
    return conv;
  }
};

} // namespace proxddp
