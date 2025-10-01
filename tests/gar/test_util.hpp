/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "aligator/gar/lqr-problem.hpp"
#include <random>

ALIGATOR_DYNAMIC_TYPEDEFS(double);
using problem_t = aligator::gar::LqrProblemTpl<double>;
using knot_t = aligator::gar::LqrKnotTpl<double>;

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

struct knot_gen_opts_t {
  uint nx;
  uint nu;
  uint nc = 0;
  uint nth = 0;
  bool singular = false;
  uint nx2 = nx;
};

knot_t generateKnot(knot_gen_opts_t opts,
                    const aligator::polymorphic_allocator &alloc = {});

problem_t generateLqProblem(const ConstVectorRef &x0, uint horz, uint nx,
                            uint nu, uint nth = 0, bool singular = true,
                            const aligator::polymorphic_allocator &alloc = {});

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

/// @brief Fill in a KKT constraint matrix and vector for the given LQ problem
/// with the given dual-regularization parameters @p mudyn and @p mueq.
/// @returns Whether the matrices were successfully allocated.
template <typename Scalar>
bool lqrDenseMatrix(const aligator::gar::LqrProblemTpl<Scalar> &problem,
                    const Scalar mueq,
                    typename aligator::math_types<Scalar>::MatrixXs &mat,
                    typename aligator::math_types<Scalar>::VectorXs &rhs) {
  const auto &knots = problem.stages;
  const size_t N = size_t(problem.horizon());

  if (!problem.isInitialized())
    return false;

  const uint nrows = lqrNumRows(problem);
  mat.conservativeResize(nrows, nrows);
  rhs.conservativeResize(nrows);
  mat.setZero();

  uint idx = 0;
  {
    const uint nc0 = problem.nc0();
    const uint nx0 = knots[0].nx;
    mat.block(nc0, 0, nx0, nc0) = problem.G0.transpose();
    mat.block(0, nc0, nc0, nx0) = problem.G0;
    mat.topLeftCorner(nc0, nc0).setZero();

    rhs.head(nc0) = problem.g0;
    idx += nc0;
  }

  for (size_t t = 0; t <= N; t++) {
    const auto &model = knots[t];
    // get block for current variables
    const uint n = model.nx + model.nu + model.nc;
    auto block = mat.block(idx, idx, n, n);
    auto rhsblk = rhs.segment(idx, n);
    auto Q = block.topLeftCorner(model.nx, model.nx);
    auto St = block.leftCols(model.nx).middleRows(model.nx, model.nu);
    auto R = block.block(model.nx, model.nx, model.nu, model.nu);
    auto C = block.bottomRows(model.nc).leftCols(model.nx);
    auto D = block.bottomRows(model.nc).middleCols(model.nx, model.nu);
    auto dual = block.bottomRightCorner(model.nc, model.nc).diagonal();
    dual.array() = -mueq;

    Q = model.Q;
    St = model.S.transpose();
    R = model.R;
    C = model.C;
    D = model.D;

    block = block.template selfadjointView<Eigen::Lower>();

    rhsblk.head(model.nx) = model.q;
    rhsblk.segment(model.nx, model.nu) = model.r;
    rhsblk.tail(model.nc) = model.d;

    // fill in dynamics
    // row contains [A; B; 0; -mu*I, E] -> nx + nu + nc + nx + nx2 cols
    if (t != N) {
      uint ncols = model.nx + model.nx2 + n;
      auto row = mat.block(idx + n, idx, model.nx, ncols);
      row.leftCols(model.nx) = model.A;
      row.middleCols(model.nx, model.nu) = model.B;
      row.middleCols(n, model.nx).setZero();
      row.rightCols(model.nx).setIdentity() *= -1;

      rhs.segment(idx + n, model.nx2) = model.f;

      auto col = mat.transpose().block(idx + n, idx, model.nx, ncols);
      col = row;

      // shift by size of block + costate size (nx2)
      idx += n + model.nx2;
    }
  }
  return true;
}

/// @copybrief lqrDenseMatrix()
template <typename Scalar>
auto lqrDenseMatrix(const aligator::gar::LqrProblemTpl<Scalar> &problem,
                    const Scalar mueq) {
  using aligator::math_types;
  decltype(auto) knots = problem.stages;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  const uint nrows = lqrNumRows(problem);

  MatrixXs mat(nrows, nrows);
  VectorXs rhs(nrows);

  if (!lqrDenseMatrix(problem, mueq, mat, rhs)) {
    ALIGATOR_WARNING("lqrDenseMatrix",
                     "{:s} WARNING! Problem was not initialized.",
                     __FUNCTION__);
  }
  return std::make_pair(mat, rhs);
}
