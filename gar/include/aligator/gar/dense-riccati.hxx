/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include "dense-riccati.hpp"
#include "lqr-problem.hpp"
#include "aligator/utils/mpc-util.hpp"
#include "aligator/tracy.hpp"

namespace aligator::gar {

template <typename Scalar>
void RiccatiSolverDense<Scalar>::init_factor(const LqrKnotTpl<Scalar> &knot) {
  std::array<long, 4> dims = {knot.nu, knot.nc, knot.nx2, knot.nx2};
  long ntot = dims[0] + dims[1] + dims[2] + dims[3];
  kkts.emplace_back(BlkMat44::Zero(dims, dims));
  ffs.emplace_back(BlkVec4::Zero(dims, {1}));
  fbs.emplace_back(BlkRowMat41::Zero(dims, {knot.nx}));
  fts.emplace_back(BlkRowMat41::Zero(dims, {knot.nth}));
  ldls.emplace_back(ntot);
}

template <typename Scalar>
RiccatiSolverDense<Scalar>::RiccatiSolverDense(
    const LqrProblemTpl<Scalar> &problem)
    : Base(), problem_(&problem) {
  initialize();
}

template <typename Scalar> void RiccatiSolverDense<Scalar>::initialize() {
  auto N = (uint)problem_->horizon();
  const auto &stages = problem_->stages;
  Pxx.resize(N + 1);
  Pxt.resize(N + 1);
  Ptt.resize(N + 1);
  px.resize(N + 1);
  pt.resize(N + 1);
  for (uint i = 0; i <= N; i++) {
    uint nx = stages[i].nx;
    uint nth = stages[i].nth;
    Pxx[i].setZero(nx, nx);
    Pxt[i].setZero(nx, nth);
    Ptt[i].setZero(nth, nth);
    px[i].setZero(nx);
    pt[i].setZero(nth);
    this->init_factor(stages[i]);
  }

  uint nx0 = stages[0].nx;
  uint nth = stages[0].nth;
  std::array<long, 2> dims0 = {nx0, problem_->nc0()};
  kkt0 = {decltype(kkt0.mat)(dims0, dims0), decltype(kkt0.ff)(dims0, {1}),
          decltype(kkt0.fth)(dims0, {nth}),
          Eigen::BunchKaufman<MatrixXs>(nx0 + problem_->nc0())};
  thGrad.setZero(nth);
  thHess.setZero(nth, nth);
}

template <typename Scalar>
bool RiccatiSolverDense<Scalar>::backward(const Scalar mudyn,
                                          const Scalar mueq) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  const auto &stages = problem_->stages;

  const uint N = (uint)problem_->horizon();
  {
    const KnotType &knot = stages[N];
    // FactorData &fac = datas[N];
    // fac.kkt.setZero();
    kkts[N].setZero();
    VectorRef kff = ffs[N][0];
    VectorRef zff = ffs[N][1];
    RowMatrixRef K = fbs[N].blockRow(0);
    RowMatrixRef Z = fbs[N].blockRow(1);

    // assemble last-stage kkt matrix - includes input 'u'
    kkts[N](0, 0) = knot.R;
    kkts[N](0, 1) = knot.D.transpose();
    kkts[N](1, 0) = knot.D;
    kkts[N](1, 1).diagonal().array() = -mueq;

    kff = -knot.r;
    zff = -knot.d;
    K = -knot.S.transpose();
    Z = -knot.C;

    ldls[N].compute(kkts[N].matrix());
    ldls[N].solveInPlace(ffs[N].matrix());
    ldls[N].solveInPlace(fbs[N].matrix());

    RowMatrixRef Kth = fts[N].blockRow(0);
    Kth = -knot.Gu;
    RowMatrixRef Zth = fts[N].blockRow(1);
    Zth = -knot.Gv;
    ldls[N].solveInPlace(fts[N].matrix());

    Eigen::Transpose Ct = knot.C.transpose();

    Pxx[N].noalias() = knot.Q + knot.S * K;
    Pxx[N].noalias() += Ct * Z;

    Pxt[N].noalias() = knot.Gx + K.transpose() * knot.Gu;
    Pxt[N].noalias() += Z.transpose() * knot.Gv;

    Ptt[N].noalias() = knot.Gth + knot.Gu.transpose() * Kth;
    Ptt[N].noalias() += knot.Gv.transpose() * Zth;

    px[N].noalias() = knot.q + knot.S * kff;
    px[N].noalias() += Ct * zff;

    pt[N].noalias() = knot.gamma + knot.Gu.transpose() * kff;
    pt[N].noalias() += knot.Gv.transpose() * zff;
  }

  uint i = N - 1;
  while (true) {
    const KnotType &knot = stages[i];
    // FactorData &fac = datas[i];

    kkts[i].setZero();
    kkts[i](0, 0) = knot.R;
    kkts[i](1, 0) = knot.D;
    kkts[i](0, 1) = knot.D.transpose();
    kkts[i](1, 1).diagonal().array() = -mueq;

    kkts[i](2, 0) = knot.B;
    kkts[i](0, 2) = knot.B.transpose();
    kkts[i](2, 2).diagonal().array() = -mudyn;
    kkts[i](3, 2) = knot.E.transpose();
    kkts[i](2, 3) = knot.E;
    kkts[i](3, 3) = Pxx[i + 1];

    VectorRef kff = ffs[i][0] = -knot.r;
    VectorRef zff = ffs[i][1] = -knot.d;
    VectorRef lff = ffs[i][2] = -knot.f;
    VectorRef yff = ffs[i][3] = -px[i + 1];

    RowMatrixRef K = fbs[i].blockRow(0) = -knot.S.transpose();
    RowMatrixRef Z = fbs[i].blockRow(1) = -knot.C;
    RowMatrixRef L = fbs[i].blockRow(2) = -knot.A;
    RowMatrixRef Y = fbs[i].blockRow(3);
    Y.setZero();

    RowMatrixRef Kth = fts[i].blockRow(0) = -knot.Gu;
    RowMatrixRef Zth = fts[i].blockRow(1) = -knot.Gv;
    fts[i].blockRow(2).setZero();
    RowMatrixRef Yth = fts[i].blockRow(3) = -Pxt[i + 1];

    ldls[i].compute(kkts[i].matrix());
    ldls[i].solveInPlace(ffs[i].matrix());
    ldls[i].solveInPlace(fbs[i].matrix());
    ldls[i].solveInPlace(fts[i].matrix());

    Eigen::Transpose At = knot.A.transpose();
    Eigen::Transpose Ct = knot.C.transpose();
    Pxx[i].noalias() = knot.Q + knot.S * K;
    Pxx[i].noalias() += Ct * Z;
    Pxx[i].noalias() += At * L;

    Pxt[i] = knot.Gx;
    Pxt[i].noalias() += K.transpose() * knot.Gu;
    Pxt[i].noalias() += Z.transpose() * knot.Gv;
    Pxt[i].noalias() += Y.transpose() * Pxt[i + 1];

    Ptt[i] = knot.Gth;
    Ptt[i].noalias() += Kth.transpose() * knot.Gu;
    Ptt[i].noalias() += Zth.transpose() * knot.Gv;
    Ptt[i].noalias() += Yth.transpose() * Pxt[i + 1];

    px[i].noalias() = knot.q + knot.S * kff;
    px[i].noalias() += Ct * zff;
    px[i].noalias() += At * lff;

    pt[i].noalias() = knot.gamma + knot.Gu.transpose() * kff;
    pt[i].noalias() += knot.Gv.transpose() * zff;
    pt[i].noalias() += Pxt[i + 1].transpose() * yff;

    if (i == 0)
      break;
    i--;
  }

  // initial stage
  kkt0.mat.setZero();
  kkt0.mat(0, 0) = Pxx[0];
  kkt0.mat(0, 1) = problem_->G0.transpose();
  kkt0.mat(1, 0) = problem_->G0;
  kkt0.mat(1, 1).diagonal().array() = -mudyn;
  kkt0.ldl.compute(kkt0.mat.matrix());

  kkt0.ff[0] = -px[0];
  kkt0.ff[1] = -problem_->g0;

  kkt0.fth.blockRow(0) = -Pxt[0];
  kkt0.fth.blockRow(1).setZero();

  kkt0.ldl.solveInPlace(kkt0.ff.matrix());
  kkt0.ldl.solveInPlace(kkt0.fth.matrix());

  thGrad.noalias() = pt[0] + Pxt[0].transpose() * kkt0.ff[0];
  thHess.noalias() = Ptt[0] + Pxt[0].transpose() * kkt0.fth.blockRow(0);

  return true;
}

template <typename Scalar>
bool RiccatiSolverDense<Scalar>::forward(
    std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
    std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
    const std::optional<ConstVectorRef> &theta_) const {
  ALIGATOR_NOMALLOC_SCOPED;
  xs[0] = kkt0.ff[0];
  lbdas[0] = kkt0.ff[1];
  if (theta_.has_value()) {
    xs[0].noalias() += kkt0.fth.blockRow(0) * theta_.value();
    lbdas[0].noalias() += kkt0.fth.blockRow(1) * theta_.value();
  }

  uint N = (uint)problem_->horizon();
  assert(xs.size() == N + 1);
  assert(us.size() >= N);
  assert(vs.size() == N + 1);
  assert(lbdas.size() == N + 1);
  for (uint i = 0; i <= N; i++) {
    // const FactorData &d = datas[i];
    const KnotType &model = problem_->stages[i];
    ConstVectorRef kff = ffs[i][0];
    ConstVectorRef zff = ffs[i][1];
    ConstVectorRef lff = ffs[i][2];
    ConstVectorRef yff = ffs[i][3];

    ConstRowMatrixRef K = fbs[i].blockRow(0);
    ConstRowMatrixRef Z = fbs[i].blockRow(1);
    ConstRowMatrixRef Lfb = fbs[i].blockRow(2);
    ConstRowMatrixRef Yfb = fbs[i].blockRow(3);

    ConstRowMatrixRef Kth = fts[i].blockRow(0);
    ConstRowMatrixRef Zth = fts[i].blockRow(1);
    ConstRowMatrixRef Lth = fts[i].blockRow(2);
    ConstRowMatrixRef Yth = fts[i].blockRow(3);

    if (model.nu > 0)
      us[i].noalias() = kff + K * xs[i];
    vs[i].noalias() = zff + Z * xs[i];
    if (theta_.has_value()) {
      if (model.nu > 0)
        us[i].noalias() += Kth * theta_.value();
      vs[i].noalias() += Zth * theta_.value();
    }

    if (i == N)
      break;
    lbdas[i + 1].noalias() = lff + Lfb * xs[i];
    xs[i + 1].noalias() = yff + Yfb * xs[i];
    if (theta_.has_value()) {
      lbdas[i + 1].noalias() += Lth * theta_.value();
      xs[i + 1].noalias() += Yth * theta_.value();
    }
  }
  return true;
}

template <typename Scalar>
void RiccatiSolverDense<Scalar>::cycleAppend(const KnotType &knot) {
  auto N = (uint)problem_->horizon();

  rotate_vec_left(kkts, 0, 1);
  rotate_vec_left(ffs, 0, 1);
  rotate_vec_left(fbs, 0, 1);
  rotate_vec_left(fts, 0, 1);
  rotate_vec_left(ldls, 0, 1);

  rotate_vec_left(Pxx, 0, 1);
  rotate_vec_left(Pxt, 0, 1);
  rotate_vec_left(Ptt, 0, 1);
  rotate_vec_left(px, 0, 1);
  rotate_vec_left(pt, 0, 1);

  uint nx = knot.nx;
  uint nth = knot.nth;
  std::array<long, 4> dims = {knot.nu, knot.nc, knot.nx2, knot.nx2};
  long ntot = dims[0] + dims[1] + dims[2] + dims[3];
  // resize last ldl
  kkts[N - 1] = BlkMat44::Zero(dims, dims);
  ffs[N - 1] = BlkVec4::Zero(dims, {1});
  fbs[N - 1] = BlkRowMat41::Zero(dims, {nx});
  fts[N - 1] = BlkRowMat41::Zero(dims, {nth});
  using ldl_t = std::decay_t<decltype(ldls[0])>;
  ldls[N - 1] = ldl_t(ntot);

  Pxx[N - 1].setZero(nx, nx);
  Pxt[N - 1].setZero(nx, nth);
  Ptt[N - 1].setZero(nth, nth);
  px[N - 1].setZero(nx);
  pt[N - 1].setZero(nth);
};

} // namespace aligator::gar
