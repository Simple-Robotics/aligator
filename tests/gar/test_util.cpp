/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "./test_util.hpp"
#include "aligator/gar/utils.hpp"

std::mt19937 normal_unary_op::rng{42};

/// Generate a Wishart-distributed matrix in @p n dimensions with @p p DoF
MatrixXs sampleWishartDistributedMatrix(uint n, uint p) {
  MatrixXs root = MatrixXs::NullaryExpr(n, p, normal_unary_op{});
  return root * root.transpose();
};

knot_t generate_knot(uint nx, uint nu, uint nth, bool singular) {
  uint wishartDof = nx + nu + 1;
  knot_t out(nx, nu, 0, nx, nth);

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
  out.E = out.E.NullaryExpr(nx, nx, normal_unary_op{});
  out.E *= 1000;
  out.f = VectorXs::NullaryExpr(nx, normal_unary_op{});

  if (nth > 0) {
    out.Gx = MatrixXs::NullaryExpr(nx, nth, normal_unary_op{});
    out.Gu = MatrixXs::NullaryExpr(nu, nth, normal_unary_op{});
    out.Gth = sampleWishartDistributedMatrix(nth, nth + 2);
    out.gamma = VectorXs::NullaryExpr(nth, normal_unary_op{});
  }

  return out;
}

problem_t generate_problem(const ConstVectorRef &x0, uint horz, uint nx,
                           uint nu, uint nth) {
  assert(x0.size() == nx);

  problem_t::KnotVector knots;
  knots.reserve(horz + 1);

  auto knb = generate_knot(nx, nu, nth, true);
  for (uint i = 0; i < horz; i++) {
    knots.push_back(knb);
  }
  knots.push_back(generate_knot(nx, 0, nth)); // terminal node

  problem_t prob(std::move(knots), nx);
  prob.g0 = -x0;
  prob.G0.setIdentity();
  return prob;
}

auto fmt::formatter<KktError>::format(const KktError &err,
                                      format_context &ctx) const
    -> format_context::iterator {
  std::string s = fmt::format(
      "{{ max: {:.3e}, dual: {:.3e}, cstr: {:.3e}, dyn: {:.3e} }}\n", err.max,
      err.dual, err.cstr, err.dyn);
  return formatter<std::string>::format(s, ctx);
}

KktError computeKktError(const problem_t &problem, const VectorOfVectors &xs,
                         const VectorOfVectors &us, const VectorOfVectors &vs,
                         const VectorOfVectors &lbdas,
                         const std::optional<ConstVectorRef> &theta_,
                         const double mudyn, const double mueq) {
  auto r = aligator::gar::lqrComputeKktError(problem, xs, us, vs, lbdas, mudyn,
                                             mueq, theta_, true);
  return {r[0], r[1], r[2]};
}
