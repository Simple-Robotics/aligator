/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/gar/riccati.hpp"
#include "aligator/gar/helpers.hpp"

ALIGATOR_DYNAMIC_TYPEDEFS(double);
using vecvec_t = std::vector<VectorXs>;
using prox_riccati_t = aligator::gar::ProximalRiccatiSolver<double>;
using problem_t = aligator::gar::LQRProblemTpl<double>;
using knot_t = aligator::gar::LQRKnotTpl<double>;
using aligator::math::infty_norm;

struct KktError {
  // dynamics error
  double dyn;
  // constraint error
  double cstr;
  double dual;
};

inline KktError compute_kkt_error(const problem_t &problem, const vecvec_t &xs,
                                  const vecvec_t &us, const vecvec_t &vs,
                                  const vecvec_t &lbdas) {
  uint N = (uint)problem.horizon();

  double dynErr = 0.;
  double cstErr = 0.;
  double dualErr = 0.;

  VectorXs _dyn;
  VectorXs _cst;
  VectorXs _gx;
  VectorXs _gu;

  // initial stage
  {
    _dyn = problem.g0 + problem.G0 * xs[0];
    dynErr = std::max(dynErr, infty_norm(_dyn));
  }
  for (uint t = 0; t <= N; t++) {
    const knot_t &kn = problem.stages[t];
    auto _Str = kn.S.transpose();

    _cst = kn.C * xs[t] + kn.D * us[t] + kn.d;

    _gx = kn.Q * xs[t] + kn.S * us[t] + kn.q + kn.C.transpose() * vs[t];
    _gu = _Str * xs[t] + kn.R * us[t] + kn.r + kn.D.transpose() * vs[t];

    if (t == 0) {
      _gx += problem.G0.transpose() * lbdas[0];
    } else {
      auto Et = problem.stages[t - 1].E.transpose();
      _gx += Et * lbdas[t];
    }

    if (t < N) {
      _dyn = kn.A * xs[t] + kn.B * us[t] + kn.f + kn.E * xs[t + 1];
      _gx += kn.A.transpose() * lbdas[t + 1];
      _gu += kn.B.transpose() * lbdas[t + 1];

      dynErr = std::max(dynErr, infty_norm(_dyn));
    }

    dualErr = std::max(dualErr, infty_norm(_gx));
    dualErr = std::max(dualErr, infty_norm(_gu));
    cstErr = std::max(cstErr, infty_norm(_cst));
  }

  return {dynErr, cstErr, dualErr};
}

inline double max_kkt_error(KktError error) {
  return std::max({error.dyn, error.cstr, error.dual});
}
