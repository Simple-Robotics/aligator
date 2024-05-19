/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/gar/proximal-riccati.hpp"
#include <random>

ALIGATOR_DYNAMIC_TYPEDEFS(double);
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
  double max = std::max({dyn, cstr, dual});
};

inline void printKktError(const KktError &err,
                          const std::string &msg = "Max KKT error") {
  fmt::print("{}: {:.3e}\n", msg, err.max);
  fmt::print("> dual: {:.3e}, cstr: {:.3e}, dyn: {:.3e}\n", err.dual, err.cstr,
             err.dyn);
}

KktError
computeKktError(const problem_t &problem, const VectorOfVectors &xs,
                const VectorOfVectors &us, const VectorOfVectors &vs,
                const VectorOfVectors &lbdas,
                const std::optional<ConstVectorRef> &theta = std::nullopt,
                const double mudyn = 0., const double mueq = 0.);

inline KktError computeKktError(const problem_t &problem,
                                const VectorOfVectors &xs,
                                const VectorOfVectors &us,
                                const VectorOfVectors &vs,
                                const VectorOfVectors &lbdas,
                                const double mudyn, const double mueq) {
  return computeKktError(problem, xs, us, vs, lbdas, std::nullopt, mudyn, mueq);
}

struct normal_unary_op {
  static std::mt19937 rng;
  // underlying normal distribution
  mutable std::normal_distribution<double> gen;

  normal_unary_op(double stddev = 1.0) : gen(0.0, stddev) {}

  double operator()() const { return gen(rng); }
};

MatrixXs sampleWishartDistributedMatrix(uint n, uint p);

problem_t generate_problem(const ConstVectorRef &x0, uint horz, uint nx,
                           uint nu, uint nth = 0);

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
