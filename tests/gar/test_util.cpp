/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "./test_util.hpp"
#include "aligator/gar/block-tridiagonal-solver.hpp"

std::mt19937 normal_unary_op::rng{};

static KktError
compute_kkt_error_impl(const problem_t &problem, const VectorOfVectors &xs,
                       const VectorOfVectors &us, const VectorOfVectors &vs,
                       const VectorOfVectors &lbdas,
                       const boost::optional<ConstVectorRef> &theta_) {
  uint N = (uint)problem.horizon();

  double dynErr = 0.;
  double cstErr = 0.;
  double dualErr = 0.;
  double dNorm;
  double thNorm;

  VectorXs _dyn;
  VectorXs _cst;
  VectorXs _gx;
  VectorXs _gu;
  VectorXs _gt;

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

    fmt::print("[t={: >2d}] ", t);
    _gx.setZero(kn.nx);
    _gu.setZero(kn.nu);
    _gt.setZero(kn.nth);

    _cst = kn.C * xs[t] + kn.d;
    _gx.noalias() = kn.q + kn.Q * xs[t] + kn.C.transpose() * vs[t];
    _gu.noalias() = kn.r + _Str * xs[t] + kn.D.transpose() * vs[t];

    if (kn.nu > 0) {
      _cst.noalias() += kn.D * us[t];
      _gx.noalias() += kn.S * us[t];
      _gu.noalias() += kn.R * us[t];
    }

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

    if (theta_.has_value()) {
      ConstVectorRef th = theta_.value();
      _gx.noalias() += kn.Gx * th;
      _gu.noalias() += kn.Gu * th;
      _gt = kn.gamma;
      _gt.noalias() += kn.Gx.transpose() * xs[t];
      if (kn.nu > 0)
        _gt.noalias() += kn.Gu.transpose() * us[t];
      _gt.noalias() += kn.Gth * th;
      thNorm = infty_norm(_gt);
      fmt::print("|gt| = {:.3e} | ", thNorm);
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

/// Generate a Wishart-distributed matrix in @p n dimensions with @p p DoF
MatrixXs wishart_dist_matrix(uint n, uint p) {
  MatrixXs root = MatrixXs::NullaryExpr(n, p, normal_unary_op{});
  return root * root.transpose();
};

problem_t generate_problem(const ConstVectorRef &x0, uint horz, uint nx,
                           uint nu) {
  assert(x0.size() == nx);

  std::vector<knot_t> knots;
  uint wishartDof = nx + nu + 1;

  auto gen = [&](uint nu) {
    knot_t out(nx, nu, 0);

    MatrixXs _qsr = wishart_dist_matrix(nx + nu, wishartDof);

    out.Q = _qsr.topLeftCorner(nx, nx);
    out.S = _qsr.topRightCorner(nx, nu);
    out.R = _qsr.bottomRightCorner(nu, nu);
    out.q.head(nx) = VectorXs::NullaryExpr(nx, normal_unary_op{});
    out.r.head(nu) = VectorXs::NullaryExpr(nu, normal_unary_op{});

    out.A.setRandom();
    out.B.setRandom();
    out.E.setIdentity();
    out.E *= -1;
    out.f.head(nx) = VectorXs::NullaryExpr(nx, normal_unary_op{});

    return out;
  };

  auto knb = gen(nu);

  for (uint i = 0; i < horz; i++) {
    knots.push_back(knb);
  }
  auto knl = gen(0);
  knots.push_back(knl); // terminal node

  problem_t prob(knots, nx);
  prob.g0 = -x0;
  prob.G0.setIdentity();
  return prob;
}

KktError compute_kkt_error(const problem_t &problem, const VectorOfVectors &xs,
                           const VectorOfVectors &us, const VectorOfVectors &vs,
                           const VectorOfVectors &lbdas) {
  return compute_kkt_error_impl(problem, xs, us, vs, lbdas, boost::none);
}

KktError compute_kkt_error(const problem_t &problem, const VectorOfVectors &xs,
                           const VectorOfVectors &us, const VectorOfVectors &vs,
                           const VectorOfVectors &lbdas,
                           const ConstVectorRef &theta) {
  return compute_kkt_error_impl(problem, xs, us, vs, lbdas, theta);
}

MatrixXs block_tridiag_to_dense(std::vector<MatrixXs> const &subdiagonal,
                                std::vector<MatrixXs> const &diagonal,
                                std::vector<MatrixXs> const &superdiagonal) {
  if (!aligator::gar::internal::check_block_tridiag(subdiagonal, diagonal,
                                                    superdiagonal)) {
    throw std::invalid_argument("Wrong lengths");
  }

  const size_t N = subdiagonal.size();
  Eigen::Index dim = 0;
  for (size_t i = 0; i <= N; i++) {
    dim += diagonal[i].cols();
  }

  MatrixXs out(dim, dim);
  out.setZero();
  Eigen::Index i0 = 0;
  for (size_t i = 0; i <= N; i++) {
    Eigen::Index d = diagonal[i].cols();
    out.block(i0, i0, d, d) = diagonal[i];

    if (i != N) {
      Eigen::Index dn = superdiagonal[i].cols();
      out.block(i0, i0 + d, d, dn) = superdiagonal[i];
      out.block(i0 + d, i0, dn, d) = subdiagonal[i];
    }

    i0 += d;
  }
  return out;
}
