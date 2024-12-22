/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/gar/lqr-problem.hpp"
#include <random>

ALIGATOR_DYNAMIC_TYPEDEFS(double);
using problem_t = aligator::gar::LqrProblemTpl<double>;
using knot_t = aligator::gar::LqrKnotTpl<double>;
using aligator::math::infty_norm;

struct KktError {
  // dynamics error
  double dyn;
  // constraint error
  double cstr;
  double dual;
  double max = std::max({dyn, cstr, dual});
};

template <> struct fmt::formatter<KktError> : formatter<std::string_view> {
  auto format(const KktError &err, format_context &ctx) const
      -> format_context::iterator;
};

KktError
computeKktError(const problem_t &problem, const VectorOfVectors &xs,
                const VectorOfVectors &us, const VectorOfVectors &vs,
                const VectorOfVectors &lbdas,
                const std::optional<ConstVectorRef> &theta = std::nullopt,
                const double mudyn = 0., const double mueq = 0.,
                bool verbose = true);

inline KktError
computeKktError(const problem_t &problem, const VectorOfVectors &xs,
                const VectorOfVectors &us, const VectorOfVectors &vs,
                const VectorOfVectors &lbdas, const double mudyn,
                const double mueq, bool verbose = true) {
  return computeKktError(problem, xs, us, vs, lbdas, std::nullopt, mudyn, mueq,
                         verbose);
}

struct normal_unary_op {
  static std::mt19937 rng;
  // underlying normal distribution
  mutable std::normal_distribution<double> gen;

  static void set_rng(size_t sd) { rng = std::mt19937{sd}; }

  normal_unary_op(double stddev = 1.0) : gen(0.0, stddev) {}
  double operator()() const { return gen(rng); }
};

MatrixXs sampleWishartDistributedMatrix(uint n, uint p);

knot_t generate_knot(uint nx, uint nu, uint nth, bool singular = false,
                     const aligator::polymorphic_allocator &alloc = {});

inline knot_t generate_knot(uint nx, uint nu, uint nth,
                            const aligator::polymorphic_allocator &alloc) {
  return generate_knot(nx, nu, nth, false, alloc);
}

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
