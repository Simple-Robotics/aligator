/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "./util.hpp"
#include <random>

struct normal_unary_op {
  static std::mt19937 rng;
  // underlying normal distribution
  mutable std::normal_distribution<double> gen;

  normal_unary_op(double stddev = 1.0) : gen(0.0, stddev) {}

  double operator()() const { return gen(rng); }
};

std::mt19937 normal_unary_op::rng{};

KktError compute_kkt_error(const problem_t &problem, const vecvec_t &xs,
                           const vecvec_t &us, const vecvec_t &vs,
                           const vecvec_t &lbdas,
                           const boost::optional<ConstVectorRef> &theta_) {
  uint N = (uint)problem.horizon();

  double dynErr = 0.;
  double cstErr = 0.;
  double dualErr = 0.;
  double dNorm;

  VectorXs _dyn;
  VectorXs _cst;
  VectorXs _gx;
  VectorXs _gu;

  // initial stage
  {
    _dyn = problem.g0 + problem.G0 * xs[0];
    dNorm = infty_norm(_dyn);
    dynErr = std::max(dynErr, dNorm);
    fmt::print(" |d| = {:.3e} \n", dNorm);
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

      dNorm = infty_norm(_dyn);
      fmt::print(" |d| = {:.3e} | ", dNorm);
      dynErr = std::max(dynErr, dNorm);
    }

    double gxNorm = infty_norm(_gx);
    double guNorm = infty_norm(_gu);
    double cstNorm = infty_norm(_cst);
    fmt::print("|gx| = {:.3e} | |gu| = {:.3e} | |cst| = {:.3e}\n", gxNorm,
               guNorm, cstNorm);

    dualErr = std::max({dualErr, gxNorm, guNorm});
    cstErr = std::max(cstErr, cstNorm);
  }

  return {dynErr, cstErr, dualErr, std::max({dynErr, cstErr, dualErr})};
}

/// Generate a Wishart-distributed matrix
auto wishart_dist_matrix(uint n, uint p) {
  MatrixXs root = MatrixXs::NullaryExpr(n, p, normal_unary_op{});
  return MatrixXs(root * root.transpose());
};

problem_t generate_problem(const ConstVectorRef &x0, uint horz, uint nx,
                           uint nu) {
  assert(x0.size() == nx);

  std::vector<knot_t> knots;

  auto gen = [&]() {
    knot_t out(nx, nu, 0);

    out.Q = wishart_dist_matrix(nx, 1000);
    out.R = wishart_dist_matrix(nu, 1000);
    out.q.head(nx) = VectorXs::NullaryExpr(nx, normal_unary_op{});
    out.r.head(nu) = VectorXs::NullaryExpr(nu, normal_unary_op{});

    out.A.setRandom();
    out.B.setRandom();
    out.E.setIdentity();
    out.E *= -1;
    // out.f.head(nx) = VectorXs::NullaryExpr(nx, normal_unary_op{});

    return out;
  };

  auto knb = gen();

  for (uint i = 0; i <= horz; i++) {
    knots.emplace_back(knb);
  }

  problem_t prob(knots, nx);
  prob.g0 = -x0;
  prob.G0.setIdentity();
  return prob;
}
