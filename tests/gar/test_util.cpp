/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "./test_util.hpp"
#include "aligator/gar/block-tridiagonal-solver.hpp"
#include "aligator/gar/utils.hpp"

std::mt19937 normal_unary_op::rng{};

static KktError
compute_kkt_error_impl(const problem_t &problem, const VectorOfVectors &xs,
                       const VectorOfVectors &us, const VectorOfVectors &vs,
                       const VectorOfVectors &lbdas,
                       const std::optional<ConstVectorRef> &theta_) {
  auto r = aligator::gar::lqrComputeKktError(problem, xs, us, vs, lbdas, 0., 0.,
                                             theta_);
  return {r[0], r[1], r[2], std::max({r[0], r[1], r[2]})};
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
  return compute_kkt_error_impl(problem, xs, us, vs, lbdas, std::nullopt);
}

KktError compute_kkt_error(const problem_t &problem, const VectorOfVectors &xs,
                           const VectorOfVectors &us, const VectorOfVectors &vs,
                           const VectorOfVectors &lbdas,
                           const ConstVectorRef &theta) {
  return compute_kkt_error_impl(problem, xs, us, vs, lbdas, theta);
}
