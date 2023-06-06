#pragma once

#include "proxddp/fwd.hpp"

#include <Eigen/LU>

namespace proxddp {

/// @brief  Newton-Raphson procedure, e.g. to compute forward dynamics from
/// implicit functions.
template <typename Scalar> struct NewtonRaphson {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Manifold = ManifoldAbstractTpl<Scalar>;

  struct Options {
    Scalar alpha_min = 1e-4;
    Scalar ls_beta = 0.7071;
    Scalar armijo_c1 = 1e-2;
  };

  template <typename Fun, typename JacFun>
  static bool run(const Manifold &space, Fun &&fun, JacFun &&jac_fun,
                  const ConstVectorRef &xinit, VectorRef xout, VectorRef f0,
                  VectorRef dx, MatrixRef Jf0, Scalar eps = 1e-6,
                  std::size_t max_iters = 1000, Options options = Options{}) {

    xout = xinit;

    fun(xout, f0);

    // workspace
    VectorXs dx_ls = dx;
    VectorXs xcand = xout;

    Scalar err = f0.norm();
    std::size_t iter = 0;
    while (true) {

      if (err <= eps) {
        return true;
      } else if (iter >= max_iters) {
        return false;
      }

      jac_fun(xout, Jf0);
      dx = Jf0.lu().solve(-f0);

      // linesearch
      Scalar alpha = 1.;
      while (alpha > options.alpha_min) {
        dx_ls = alpha * dx; // avoid malloc in ls
        space.integrate(xout, dx_ls, xcand);
        fun(xcand, f0);
        Scalar cand_err = f0.norm();
        if (cand_err <= (1. - options.armijo_c1) * err) {
          xout = xcand;
          err = cand_err;
          break;
        }
        alpha *= options.ls_beta;
      }

      iter++;
    }
  }
};

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/utils/newton-raphson.txx"
#endif
