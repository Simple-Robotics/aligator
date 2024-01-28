/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./lqr-problem.hpp"

namespace aligator {
namespace gar {

template <typename Scalar>
auto lqrComputeKktError(
    const LQRProblemTpl<Scalar> &problem,
    const typename math_types<Scalar>::VectorOfVectors &xs,
    const typename math_types<Scalar>::VectorOfVectors &us,
    const typename math_types<Scalar>::VectorOfVectors &vs,
    const typename math_types<Scalar>::VectorOfVectors &lbdas,
    const Scalar mudyn, const Scalar mueq,
    const std::optional<typename math_types<Scalar>::ConstVectorRef> &theta_,
    bool verbose = false) {
  fmt::print("[{}] ", __func__);
  uint N = (uint)problem.horizon();
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using KnotType = LQRKnotTpl<Scalar>;

  Scalar dynErr = 0.;
  Scalar cstErr = 0.;
  Scalar dualErr = 0.;
  Scalar dNorm;
  Scalar thNorm;

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
      thNorm = math::infty_norm(_gt);
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
                    typename math_types<Scalar>::VectorXs &rhs) {
  using knot_t = LQRKnotTpl<Scalar>;
  const std::vector<knot_t> &knots = problem.stages;
  size_t N = (size_t)problem.horizon();

  if (!problem.isInitialized())
    return false;

  mat.setZero();

  uint idx = 0;
  {
    uint nc0 = problem.nc0();
    uint nx0 = problem.stages[0].nx;
    mat.block(nc0, 0, nx0, nc0) = problem.G0.transpose();
    mat.block(0, nc0, nc0, nx0) = problem.G0;
    mat.topLeftCorner(nc0, nc0).diagonal().setConstant(-mudyn);

    rhs.head(nc0) = problem.g0;
    idx += nc0;
  }

  for (size_t t = 0; t <= N; t++) {
    const knot_t &model = knots[t];
    // get block for current variables
    uint n = model.nx + model.nu + model.nc;
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
      row.middleCols(n, model.nx).diagonal().array() = -mudyn;
      row.rightCols(model.nx) = model.E;

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
auto lqrDenseMatrix(const LQRProblemTpl<Scalar> &problem, Scalar mudyn,
                    Scalar mueq) {

  decltype(auto) knots = problem.stages;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using knot_t = LQRKnotTpl<Scalar>;
  uint nc0 = problem.nc0();
  size_t N = knots.size() - 1UL;
  uint nrows = nc0;
  for (size_t t = 0; t <= N; t++) {
    const knot_t &model = knots[t];
    nrows += model.nx + model.nu + model.nc;
    if (t != N)
      nrows += model.nx;
  }

  MatrixXs mat(nrows, nrows);
  VectorXs rhs(nrows);

  if (!lqrDenseMatrix(problem, mudyn, mueq, mat, rhs)) {
    ALIGATOR_RUNTIME_ERROR("Problem was not initialized.");
  }
  return std::make_pair(mat, rhs);
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
extern template auto lqrComputeKktError<context::Scalar>(
    const LQRProblemTpl<context::Scalar> &, const context::VectorOfVectors &,
    const context::VectorOfVectors &, const context::VectorOfVectors &,
    const context::VectorOfVectors &, const context::Scalar,
    const context::Scalar, const std::optional<context::ConstVectorRef> &,
    bool);
extern template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
#endif

} // namespace gar
} // namespace aligator
