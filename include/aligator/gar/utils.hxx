#pragma once
#include "utils.hpp"

namespace aligator::gar {

template <typename Scalar>
void lqrCreateSparseMatrix(const LqrProblemTpl<Scalar> &problem,
                           const Scalar mueq, Eigen::SparseMatrix<Scalar> &mat,
                           Eigen::Matrix<Scalar, -1, 1> &rhs, bool update) {
  using Eigen::Index;
  const uint nrows = lqrNumRows(problem);
  const auto &knots = problem.stages;

  if (!update) {
    mat.conservativeResize(nrows, nrows);
    mat.setZero();
  }

  rhs.conservativeResize(nrows);
  rhs.setZero();
  const size_t N = size_t(problem.horizon());
  uint idx = 0;
  {
    uint nc0 = problem.nc0();
    rhs.head(nc0) = problem.g0;
    helpers::sparseAssignDenseBlock(0, nc0, problem.G0, mat, update);
    helpers::sparseAssignDenseBlock(nc0, 0, problem.G0.transpose(), mat,
                                    update);
    idx += nc0;
  }

  for (size_t t = 0; t <= N; t++) {
    const LqrKnotTpl<Scalar> &model = knots[t];
    const uint n = model.nx + model.nu + model.nc;
    // get block for current variables
    auto rhsblk = rhs.segment(idx, n);

    rhsblk.head(model.nx) = model.q;
    rhsblk.segment(model.nx, model.nu) = model.r;
    rhsblk.tail(model.nc) = model.d;

    // fill-in Q block
    helpers::sparseAssignDenseBlock(idx, idx, model.Q, mat, update);
    const Index i0 = idx + model.nx; // u
    // S block
    helpers::sparseAssignDenseBlock(i0, idx, model.S.transpose(), mat, update);
    helpers::sparseAssignDenseBlock(idx, i0, model.S, mat, update);
    // R block
    helpers::sparseAssignDenseBlock(i0, i0, model.R, mat, update);

    const Index i1 = i0 + model.nu; // v
    // C block
    helpers::sparseAssignDenseBlock(i1, idx, model.C, mat, update);
    helpers::sparseAssignDenseBlock(idx, i1, model.C.transpose(), mat, update);
    // D block
    helpers::sparseAssignDenseBlock(i1, i0, model.D, mat, update);
    helpers::sparseAssignDenseBlock(i0, i1, model.D.transpose(), mat, update);

    const Index i2 = i1 + model.nc;
    // dual block
    helpers::sparseAssignDiagonal(i1, i2, -mueq, mat, update);

    if (t != N) {
      rhs.segment(idx + n, model.nx2) = model.f;

      // A
      helpers::sparseAssignDenseBlock(i2, idx, model.A, mat, update);
      helpers::sparseAssignDenseBlock(idx, i2, model.A.transpose(), mat,
                                      update);
      // B
      helpers::sparseAssignDenseBlock(i2, i0, model.B, mat, update);
      helpers::sparseAssignDenseBlock(i0, i2, model.B.transpose(), mat, update);
      // E
      const Index i3 = i2 + model.nx2;
      helpers::sparseAssignDenseBlock(i2, i3, model.E, mat, update);
      helpers::sparseAssignDenseBlock(i3, i2, model.E.transpose(), mat, update);

      idx += n + model.nx2;
    }
  }
}

template <typename Scalar>
std::array<Scalar, 3> lqrComputeKktError(
    const LqrProblemTpl<Scalar> &problem,
    boost::span<const typename math_types<Scalar>::VectorXs> xs,
    boost::span<const typename math_types<Scalar>::VectorXs> us,
    boost::span<const typename math_types<Scalar>::VectorXs> vs,
    boost::span<const typename math_types<Scalar>::VectorXs> lbdas,
    const Scalar mudyn, const Scalar mueq,
    const std::optional<typename math_types<Scalar>::ConstVectorRef> &theta_,
    bool verbose) {
  if (verbose)
    fmt::print("[{}] ", __func__);
  uint N = (uint)problem.horizon();
  assert(xs.size() == N + 1);
  using VectorXs = typename math_types<Scalar>::VectorXs;

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
    const LqrKnotTpl<Scalar> &knot = problem.stages[t];
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

template <typename Scalar>
bool lqrDenseMatrix(const LqrProblemTpl<Scalar> &problem, Scalar mudyn,
                    Scalar mueq, typename math_types<Scalar>::MatrixXs &mat,
                    typename math_types<Scalar>::VectorXs &rhs) {
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
    mat.topLeftCorner(nc0, nc0).diagonal().setConstant(-mudyn);

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

} // namespace aligator::gar
