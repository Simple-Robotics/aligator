/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
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

template <> struct fmt::formatter<KktError> {
  constexpr auto parse(format_parse_context &ctx) const
      -> decltype(ctx.begin()) {
    return ctx.end();
  }

  auto format(const KktError &err, format_context &ctx) const
      -> decltype(ctx.out()) {
    return fmt::format_to(
        ctx.out(), "{{ max: {:.3e}, dual: {:.3e}, cstr: {:.3e}, dyn: {:.3e} }}",
        err.max, err.dual, err.cstr, err.dyn);
  }
};

KktError
computeKktError(const problem_t &problem, const VectorOfVectors &xs,
                const VectorOfVectors &us, const VectorOfVectors &vs,
                const VectorOfVectors &lbdas,
                const std::optional<ConstVectorRef> &theta = std::nullopt,
                const double mueq = 0., bool verbose = true);

inline KktError computeKktError(const problem_t &problem,
                                const VectorOfVectors &xs,
                                const VectorOfVectors &us,
                                const VectorOfVectors &vs,
                                const VectorOfVectors &lbdas, const double mueq,
                                bool verbose = true) {
  return computeKktError(problem, xs, us, vs, lbdas, std::nullopt, mueq,
                         verbose);
}

struct normal_unary_op {
  static std::mt19937 rng;
  // underlying normal distribution
  mutable std::normal_distribution<double> gen;

  normal_unary_op(double stddev = 1.0)
      : gen(0.0, stddev) {}
  static void set_seed(size_t sd) { rng.seed(sd); }

  double operator()() const { return gen(rng); }
};

MatrixXs sampleWishartDistributedMatrix(uint n, uint p);

knot_t generateKnot(uint nx, uint nu, uint nth, bool singular = false,
                    const aligator::polymorphic_allocator &alloc = {});

inline knot_t generateKnot(uint nx, uint nu, uint nth,
                           const aligator::polymorphic_allocator &alloc) {
  return generateKnot(nx, nu, nth, false, alloc);
}

problem_t generateLqProblem(const ConstVectorRef &x0, uint horz, uint nx,
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
