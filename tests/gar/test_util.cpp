/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#include "./test_util.hpp"
#include "aligator/gar/utils.hpp"

std::mt19937 normal_unary_op::rng{42};

/// Generate a Wishart-distributed matrix in @p n dimensions with @p p DoF
MatrixXs sampleWishartDistributedMatrix(uint n, uint p) {
  MatrixXs root = MatrixXs::NullaryExpr(n, p, normal_unary_op{});
  return root * root.transpose();
};

knot_t generateKnot(uint nx, uint nu, uint nth, bool singular,
                    const aligator::polymorphic_allocator &alloc) {
  uint wishartDof = nx + nu + 1;
  knot_t out(nx, nu, 0, nx, nth, alloc);
  MatrixXs _qsr = sampleWishartDistributedMatrix(nx + nu, wishartDof);

  out.Q = _qsr.topLeftCorner(nx, nx);
  out.S = _qsr.topRightCorner(nx, nu);
  if (singular) {
    auto n = nx / 2;
    out.Q.topLeftCorner(n, n).setZero();
  }
  out.R = _qsr.bottomRightCorner(nu, nu);
  out.q = VectorXs::NullaryExpr(nx, normal_unary_op{});
  out.r = VectorXs::NullaryExpr(nu, normal_unary_op{});

  out.A = MatrixXs::NullaryExpr(nx, nx, normal_unary_op{});
  out.B.setRandom();
  out.E = MatrixXs::NullaryExpr(nx, nx, normal_unary_op{});
  out.E *= 10;
  out.f = VectorXs::NullaryExpr(nx, normal_unary_op{100.});

  if (nth > 0) {
    out.Gx = MatrixXs::NullaryExpr(nx, nth, normal_unary_op{});
    out.Gu = MatrixXs::NullaryExpr(nu, nth, normal_unary_op{});
    out.Gth = sampleWishartDistributedMatrix(nth, nth + 2);
    out.gamma = VectorXs::NullaryExpr(nth, normal_unary_op{});
  }

  assert(out.get_allocator() == alloc);
  return out;
}

problem_t generateLqProblem(const ConstVectorRef &x0, uint horz, uint nx,
                            uint nu, uint nth, bool singular,
                            const aligator::polymorphic_allocator &alloc) {
  assert(x0.size() == nx);

  problem_t::KnotVector knots{alloc};
  knots.reserve(horz + 1);

  auto knb = generateKnot(nx, nu, nth, singular, alloc);
  for (uint i = 0; i < horz; i++) {
    knots.push_back(knb);
  }
  knots.push_back(generateKnot(nx, 0, nth, false, alloc));

  problem_t prob(std::move(knots), nx);
  prob.g0 = -x0;
  prob.G0.setIdentity();
  return prob;
}

KktError computeKktError(const problem_t &problem, const VectorOfVectors &xs,
                         const VectorOfVectors &us, const VectorOfVectors &vs,
                         const VectorOfVectors &lbdas,
                         const std::optional<ConstVectorRef> &theta_,
                         const double mueq, bool verbose) {
  auto r = aligator::gar::lqrComputeKktError(problem, xs, us, vs, lbdas, 0.0,
                                             mueq, theta_, verbose);
  return {r[0], r[1], r[2]};
}
