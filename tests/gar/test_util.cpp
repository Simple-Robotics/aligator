/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "./test_util.hpp"
#include "aligator/gar/block-tridiagonal-solver.hpp"
#include "aligator/gar/utils.hpp"

std::mt19937 normal_unary_op::rng{42};

/// Generate a Wishart-distributed matrix in @p n dimensions with @p p DoF
MatrixXs sampleWishartDistributedMatrix(uint n, uint p) {
  MatrixXs root = MatrixXs::NullaryExpr(n, p, normal_unary_op{});
  return root * root.transpose();
};

problem_t generate_problem(const ConstVectorRef &x0, uint horz, uint nx,
                           uint nu, uint nth) {
  assert(x0.size() == nx);

  std::vector<knot_t> knots;
  uint wishartDof = nx + nu + 1;

  auto gen = [&](uint nu) {
    knot_t out(nx, nu, 0);
    out.addParameterization(nth);

    MatrixXs _qsr = sampleWishartDistributedMatrix(nx + nu, wishartDof);

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

    out.Gx = MatrixXs::NullaryExpr(nx, nth, normal_unary_op{});
    out.Gu = MatrixXs::NullaryExpr(nu, nth, normal_unary_op{});
    out.Gth = sampleWishartDistributedMatrix(nth, nth + 2);
    out.gamma = VectorXs::NullaryExpr(nth, normal_unary_op{});

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

KktError computeKktError(const problem_t &problem, const VectorOfVectors &xs,
                         const VectorOfVectors &us, const VectorOfVectors &vs,
                         const VectorOfVectors &lbdas,
                         const std::optional<ConstVectorRef> &theta_,
                         const double mudyn, const double mueq) {
  auto r = aligator::gar::lqrComputeKktError(problem, xs, us, vs, lbdas, mudyn,
                                             mueq, theta_, true);
  return {r[0], r[1], r[2], std::max({r[0], r[1], r[2]})};
}
