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
  double max;
};

KktError compute_kkt_error(const problem_t &problem, const vecvec_t &xs,
                           const vecvec_t &us, const vecvec_t &vs,
                           const vecvec_t &lbdas);

KktError compute_kkt_error(const problem_t &problem, const vecvec_t &xs,
                           const vecvec_t &us, const vecvec_t &vs,
                           const vecvec_t &lbdas, const ConstVectorRef &theta);

MatrixXs wishart_dist_matrix(uint n, uint p);

problem_t generate_problem(const ConstVectorRef &x0, uint horz, uint nx,
                           uint nu);

template <typename T>
std::vector<T> mergeStdVectors(const std::vector<T> &v1,
                               const std::vector<T> &v2) {
  std::vector<T> out;
  for (size_t i = 0; i < v1.size(); i++) {
    out.push_back(v1[i]);
  }
  for (size_t i = 0; i < v2.size(); i++) {
    out.push_back(v2[i]);
  }
  return out;
}

/// Split a given problem into two parts
inline std::array<problem_t, 2> splitProblemInTwo(const problem_t &problem,
                                                  uint t0, double mu = 0.) {
  assert(problem.isInitialized());
  uint N = (uint)problem.horizon();
  assert(t0 < N);

  std::vector<knot_t> knots1, knots2;
  uint nx_t0 = problem.stages[t0].nx;

  for (uint i = 0; i < t0; i++)
    knots1.push_back(problem.stages[i]);

  for (uint i = t0; i <= N; i++)
    knots2.push_back(problem.stages[i]);

  knot_t kn1_last = knots1.back(); // copy

  problem_t p1(knots1, problem.nc0());
  p1.G0 = problem.G0;
  p1.g0 = problem.g0;
  p1.addParameterization(nx_t0);
  {
    knot_t &p1_last = p1.stages.back();
    p1_last.Gx = kn1_last.A.transpose();
    p1_last.Gu = kn1_last.B.transpose();
    p1_last.gamma = kn1_last.f;
    p1_last.Gth.diagonal().setConstant(-mu);
    kn1_last.A.setZero();
    kn1_last.B.setZero();
    kn1_last.f.setZero();
  }

  problem_t p2(knots2, 0);
  p2.addParameterization(nx_t0);
  {
    knot_t &p2_first = p2.stages[0];
    p2_first.Gx = kn1_last.E.transpose();
  }

  return {p1, p2};
}
