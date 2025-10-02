/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#include "./test_util.hpp"
#include "aligator/gar/utils.hpp"

/// Generate a Wishart-distributed matrix in @p n dimensions with @p p DoF
MatrixXs sampleWishartDistributedMatrix(uint n, uint p) {
  std::mt19937 rng;
  MatrixXs root = MatrixXs::NullaryExpr(n, p, normal_unary_op(rng));
  return root * root.transpose();
};

knot_t generateKnot(std::mt19937 rng, knot_gen_opts_t opts,
                    const aligator::polymorphic_allocator &alloc) {
  normal_unary_op normal_op{rng};
  auto nx = opts.nx;
  auto nu = opts.nu;
  auto nc = opts.nc;
  auto nth = opts.nth;
  uint wishartDof = nx + nu + 1;
  knot_t out(nx, nu, nc, opts.nx2, nth, alloc);
  MatrixXs _qsr = sampleWishartDistributedMatrix(nx + nu, wishartDof);
  _qsr /= std::max(nx, nu);

  out.Q = _qsr.topLeftCorner(nx, nx);
  out.S = _qsr.topRightCorner(nx, nu);
  if (opts.singular) {
    auto n = nx / 2;
    out.Q.topLeftCorner(n, n).setZero();
  }
  out.R = _qsr.bottomRightCorner(nu, nu);
  out.q.setRandom();
  out.r.setRandom();

  out.A.setRandom();
  out.B.setRandom();
  out.f = VectorXs::NullaryExpr(nx, normal_op);

  if (nc > 0) {
    out.C.setIdentity();
    out.d.setRandom();
  }

  if (nth > 0) {
    out.Gx = MatrixXs::NullaryExpr(nx, nth, normal_op);
    out.Gu = MatrixXs::NullaryExpr(nu, nth, normal_op);
    out.Gth = sampleWishartDistributedMatrix(nth, nth + 2);
    out.gamma = VectorXs::NullaryExpr(nth, normal_op);
  }

  assert(out.get_allocator() == alloc);
  return out;
}

problem_t generateLqProblem(std::mt19937 rng, const ConstVectorRef &x0,
                            uint horz, uint nx, uint nu, uint nth, uint nc,
                            bool singular,
                            const aligator::polymorphic_allocator &alloc) {
  assert(x0.size() == nx);

  problem_t::KnotVector knots{alloc};
  knots.reserve(horz + 1);

  for (uint i = 0; i < horz; i++) {
    auto knb = generateKnot(rng, {nx, nu, nc, nth, singular}, alloc);
    knots.push_back(knb);
  }
  knots.push_back(generateKnot(rng, {nx, 0, nc, nth, false}, alloc));

  problem_t prob(std::move(knots), nx);
  prob.g0 = x0;
  prob.G0.setIdentity() *= -1;
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
