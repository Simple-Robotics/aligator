/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#pragma once

#include "lqr-problem.hpp"
#include "aligator/third-party/boost/core/span.hpp"
#include <Eigen/SparseCore>

namespace aligator::gar {
namespace helpers {
/// @brief Helper to assign a dense matrix into a range of coefficients of a
/// sparse matrix.
template <typename InType, typename OutScalar>
void sparseAssignDenseBlock(Eigen::Index i0, Eigen::Index j0,
                            const Eigen::DenseBase<InType> &input,
                            Eigen::SparseMatrix<OutScalar> &out, bool update) {
  assert(i0 + input.rows() <= out.rows() && "Inconsistent rows");
  assert(j0 + input.cols() <= out.cols() && "Inconsistent cols");
  using Eigen::Index;
  for (Index i = 0; i < input.rows(); i++) {
    for (Index j = 0; j < input.cols(); j++) {
      if (update)
        out.coeffRef(i0 + i, j0 + j) = input(i, j);
      else
        out.insert(i0 + i, j0 + j) = input(i, j);
    }
  }
}

template <typename Scalar>
void sparseAssignDiagonal(Eigen::Index i0, Eigen::Index i1, Scalar value,
                          Eigen::SparseMatrix<Scalar> &out, bool update) {
  using Eigen::Index;
  assert(i0 <= i1 && "i0 should be lesser than i1. Can't assign empty range.");
  assert(i1 <= out.rows() && "Inconsistent rows");
  assert(i1 <= out.cols() && "Inconsistent cols");
  for (Index kk = i0; kk < i1; kk++) {
    if (update)
      out.coeffRef(kk, kk) = value;
    else
      out.insert(kk, kk) = value;
  }
}
} // namespace helpers

template <typename Scalar>
void lqrCreateSparseMatrix(const LQRProblemTpl<Scalar> &problem,
                           const Scalar mudyn, const Scalar mueq,
                           Eigen::SparseMatrix<Scalar> &mat,
                           Eigen::Matrix<Scalar, -1, 1> &rhs, bool update);

template <typename Scalar>
std::array<Scalar, 3> lqrComputeKktError(
    const LQRProblemTpl<Scalar> &problem,
    boost::span<const typename math_types<Scalar>::VectorXs> xs,
    boost::span<const typename math_types<Scalar>::VectorXs> us,
    boost::span<const typename math_types<Scalar>::VectorXs> vs,
    boost::span<const typename math_types<Scalar>::VectorXs> lbdas,
    const Scalar mudyn, const Scalar mueq,
    const std::optional<typename math_types<Scalar>::ConstVectorRef> &theta_,
    bool verbose = false);

template <typename Scalar>
std::array<Scalar, 3> lqrComputeKktError(
    const LQRProblemTpl<Scalar> &problem,
    boost::span<const typename math_types<Scalar>::VectorXs> xs,
    boost::span<const typename math_types<Scalar>::VectorXs> us,
    boost::span<const typename math_types<Scalar>::VectorXs> vs,
    boost::span<const typename math_types<Scalar>::VectorXs> lbdas,
    const Scalar mudyn, const Scalar mueq,
    const std::optional<typename math_types<Scalar>::ConstVectorRef> &theta_,
    bool verbose) {
  fmt::print("[{}] ", __func__);
  uint N = (uint)problem.horizon();
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using KnotType = LQRKnotTpl<Scalar>;

  Scalar dynErr = 0.;
  Scalar cstErr = 0.;
  Scalar dualErr = 0.;
  Scalar dNorm;

  VectorXs _dyn;
  VectorXs _cst;
  VectorXs _gx;
  VectorXs _gu;
  VectorXs _gt;

  // initial stage
  {
    _dyn = problem.g0 + problem.G0 * xs[0] - mudyn * lbdas[0];
    dNorm = math::infty_norm(_dyn);
    dynErr = std::max(dynErr, dNorm);
    if (verbose)
      fmt::print("d0 = {:.3e} \n", dNorm);
  }
  for (uint t = 0; t <= N; t++) {
    const KnotType &knot = problem.stages[t];
    auto _Str = knot.S.transpose();

    if (verbose)
      fmt::print("[{: >2d}] ", t);
    _gx.setZero(knot.nx);
    _gu.setZero(knot.nu);
    _gt.setZero(knot.nth);

    _cst = knot.C * xs[t] + knot.d - mueq * vs[t];
    _gx.noalias() = knot.q + knot.Q * xs[t] + knot.C.transpose() * vs[t];
    _gu.noalias() = knot.r + _Str * xs[t] + knot.D.transpose() * vs[t];

    if (knot.nu > 0) {
      _cst.noalias() += knot.D * us[t];
      _gx.noalias() += knot.S * us[t];
      _gu.noalias() += knot.R * us[t];
    }

    if (t == 0) {
      _gx += problem.G0.transpose() * lbdas[0];
    } else {
      auto Et = problem.stages[t - 1].E.transpose();
      _gx += Et * lbdas[t];
    }

    if (t < N) {
      _dyn = knot.A * xs[t] + knot.B * us[t] + knot.f + knot.E * xs[t + 1] -
             mudyn * lbdas[t + 1];
      _gx += knot.A.transpose() * lbdas[t + 1];
      _gu += knot.B.transpose() * lbdas[t + 1];

      dNorm = math::infty_norm(_dyn);
      if (verbose)
        fmt::print(" |d| = {:.3e} | ", dNorm);
      dynErr = std::max(dynErr, dNorm);
    }

    if (theta_.has_value()) {
      Eigen::Ref<const VectorXs> th = theta_.value();
      _gx.noalias() += knot.Gx * th;
      _gu.noalias() += knot.Gu * th;
      _gt = knot.gamma;
      _gt.noalias() += knot.Gx.transpose() * xs[t];
      if (knot.nu > 0)
        _gt.noalias() += knot.Gu.transpose() * us[t];
      _gt.noalias() += knot.Gth * th;
    }

    Scalar gxNorm = math::infty_norm(_gx);
    Scalar guNorm = math::infty_norm(_gu);
    Scalar cstNorm = math::infty_norm(_cst);
    if (verbose)
      fmt::print("|gx| = {:.3e} | |gu| = {:.3e} | |cst| = {:.3e}\n", gxNorm,
                 guNorm, cstNorm);

    dualErr = std::max({dualErr, gxNorm, guNorm});
    cstErr = std::max(cstErr, cstNorm);
  }

  return std::array{dynErr, cstErr, dualErr};
}

/// @brief Fill in a KKT constraint matrix and vector for the given LQ problem
/// with the given dual-regularization parameters @p mudyn and @p mueq.
/// @returns Whether the matrices were successfully allocated.
template <typename Scalar>
bool lqrDenseMatrix(const LQRProblemTpl<Scalar> &problem, Scalar mudyn,
                    Scalar mueq, typename math_types<Scalar>::MatrixXs &mat,
                    typename math_types<Scalar>::VectorXs &rhs);

/// @brief Compute the number of rows in the problem matrix.
template <typename Scalar>
uint lqrNumRows(const LQRProblemTpl<Scalar> &problem) {
  const auto &knots = problem.stages;
  const uint nc0 = problem.nc0();
  const size_t N = knots.size() - 1UL;
  uint nrows = nc0;
  for (size_t t = 0; t <= N; t++) {
    const auto &model = knots[t];
    nrows += model.nx + model.nu + model.nc;
    if (t != N)
      nrows += model.nx;
  }
  return nrows;
}

/// @copybrief lqrDenseMatrix()
template <typename Scalar>
auto lqrDenseMatrix(const LQRProblemTpl<Scalar> &problem, Scalar mudyn,
                    Scalar mueq) {

  decltype(auto) knots = problem.stages;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  const uint nrows = lqrNumRows(problem);

  MatrixXs mat(nrows, nrows);
  VectorXs rhs(nrows);

  if (!lqrDenseMatrix(problem, mudyn, mueq, mat, rhs)) {
    fmt::print("{:s} WARNING! Problem was not initialized.", __FUNCTION__);
  }
  return std::make_pair(mat, rhs);
}

/// @brief Convert dense RHS solution to its trajectory [x,u,v,lambda] solution.
template <typename Scalar>
void lqrDenseSolutionToTraj(
    const LQRProblemTpl<Scalar> &problem,
    const typename math_types<Scalar>::ConstVectorRef solution,
    std::vector<typename math_types<Scalar>::VectorXs> &xs,
    std::vector<typename math_types<Scalar>::VectorXs> &us,
    std::vector<typename math_types<Scalar>::VectorXs> &vs,
    std::vector<typename math_types<Scalar>::VectorXs> &lbdas) {
  const uint N = (uint)problem.horizon();
  xs.resize(N + 1);
  us.resize(N + 1);
  vs.resize(N + 1);
  lbdas.resize(N + 1);
  const uint nc0 = problem.nc0();

  lbdas[0] = solution.head(nc0);

  uint idx = nc0;
  for (size_t t = 0; t <= N; t++) {
    const LQRKnotTpl<Scalar> &knot = problem.stages[t];
    const uint n = knot.nx + knot.nu + knot.nc;
    auto seg = solution.segment(idx, n);
    xs[t] = seg.head(knot.nx);
    us[t] = seg.segment(knot.nx, knot.nu);
    vs[t] = seg.segment(knot.nx + knot.nu, knot.nc);
    idx += n;
    if (t < N) {
      lbdas[t + 1] = solution.segment(idx, knot.nx2);
      idx += knot.nx2;
    }
  }
}

template <typename Scalar>
auto lqrInitializeSolution(const LQRProblemTpl<Scalar> &problem) {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using knot_t = LQRKnotTpl<Scalar>;
  std::vector<VectorXs> xs;
  std::vector<VectorXs> us;
  std::vector<VectorXs> vs;
  std::vector<VectorXs> lbdas;
  const uint N = (uint)problem.horizon();

  xs.resize(N + 1);
  us.resize(N + 1);
  vs.resize(N + 1);
  lbdas.resize(N + 1);

  lbdas[0].setZero(problem.nc0());
  for (uint i = 0; i <= N; i++) {
    const knot_t &kn = problem.stages[i];
    xs[i].setZero(kn.nx);
    us[i].setZero(kn.nu);
    vs[i].setZero(kn.nc);
    if (i == N)
      break;
    lbdas[i + 1].setZero(kn.nx2);
  }
  if (problem.stages.back().nu == 0) {
    us.pop_back();
  }
  return std::make_tuple(std::move(xs), std::move(us), std::move(vs),
                         std::move(lbdas));
}

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template void lqrCreateSparseMatrix<context::Scalar>(
    const LQRProblemTpl<context::Scalar> &problem, const context::Scalar mudyn,
    const context::Scalar mueq, Eigen::SparseMatrix<context::Scalar> &mat,
    context::VectorXs &rhs, bool update);
extern template std::array<context::Scalar, 3>
lqrComputeKktError<context::Scalar>(
    const LQRProblemTpl<context::Scalar> &,
    boost::span<const context::VectorXs>, boost::span<const context::VectorXs>,
    boost::span<const context::VectorXs>, boost::span<const context::VectorXs>,
    const context::Scalar, const context::Scalar,
    const std::optional<context::ConstVectorRef> &, bool);
extern template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
#endif

} // namespace aligator::gar
