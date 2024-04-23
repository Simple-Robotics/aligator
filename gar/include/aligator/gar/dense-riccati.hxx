#pragma once

#include "dense-riccati.hpp"

namespace aligator::gar {

template <typename Scalar>
RiccatiSolverDense<Scalar>::RiccatiSolverDense(
    const LQRProblemTpl<Scalar> &problem)
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
    datas.emplace_back(init_factor(stages[i]));
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
  ZoneScoped;
  const auto &stages = problem_->stages;

  const uint N = (uint)problem_->horizon();
  {
    const KnotType &knot = stages[N];
    FactorData &fac = datas[N];
    fac.kkt.setZero();
    VectorRef kff = fac.ff[0];
    VectorRef zff = fac.ff[1];
    MatrixRef K = fac.fb.blockRow(0);
    MatrixRef Z = fac.fb.blockRow(1);

    // assemble last-stage kkt matrix - includes input 'u'
    fac.kkt(0, 0) = knot.R;
    fac.kkt(0, 1) = knot.D.transpose();
    fac.kkt(1, 0) = knot.D;
    fac.kkt(1, 1).diagonal().array() = -mueq;

    kff = -knot.r;
    zff = -knot.d;
    K = -knot.S.transpose();
    Z = -knot.C;

    fac.ldl.compute(fac.kkt.matrix());
    fac.ldl.solveInPlace(fac.ff.matrix());
    fac.ldl.solveInPlace(fac.fb.matrix());

    MatrixRef Kth = fac.fth.blockRow(0);
    Kth = -knot.Gu;
    MatrixRef Zth = fac.fth.blockRow(1);
    Zth = -knot.Gv;
    fac.ldl.solveInPlace(fac.fth.matrix());

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
    FactorData &fac = datas[i];

    fac.kkt.setZero();
    fac.kkt(0, 0) = knot.R;
    fac.kkt(1, 0) = knot.D;
    fac.kkt(0, 1) = knot.D.transpose();
    fac.kkt(1, 1).diagonal().array() = -mueq;

    fac.kkt(2, 0) = knot.B;
    fac.kkt(0, 2) = knot.B.transpose();
    fac.kkt(2, 2).diagonal().array() = -mudyn;
    fac.kkt(3, 2) = knot.E.transpose();
    fac.kkt(2, 3) = knot.E;
    fac.kkt(3, 3) = Pxx[i + 1];

    VectorRef kff = fac.ff[0] = -knot.r;
    VectorRef zff = fac.ff[1] = -knot.d;
    VectorRef lff = fac.ff[2] = -knot.f;
    VectorRef yff = fac.ff[3] = -px[i + 1];

    MatrixRef K = fac.fb.blockRow(0) = -knot.S.transpose();
    MatrixRef Z = fac.fb.blockRow(1) = -knot.C;
    MatrixRef L = fac.fb.blockRow(2) = -knot.A;
    MatrixRef Y = fac.fb.blockRow(3);
    Y.setZero();

    MatrixRef Kth = fac.fth.blockRow(0) = -knot.Gu;
    MatrixRef Zth = fac.fth.blockRow(1) = -knot.Gv;
    fac.fth.blockRow(2).setZero();
    MatrixRef Yth = fac.fth.blockRow(3) = -Pxt[i + 1];

    fac.ldl.compute(fac.kkt.matrix());
    fac.ldl.solveInPlace(fac.ff.matrix());
    fac.ldl.solveInPlace(fac.fb.matrix());
    fac.ldl.solveInPlace(fac.fth.matrix());

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
  ALIGATOR_NOMALLOC_BEGIN;
  xs[0] = kkt0.ff[0];
  lbdas[0] = kkt0.ff[1];
  if (theta_.has_value()) {
    xs[0].noalias() += kkt0.fth.blockRow(0) * theta_.value();
    lbdas[0].noalias() += kkt0.fth.blockRow(1) * theta_.value();
  }

  uint N = (uint)problem_->horizon();
  assert(xs.size() == N + 1);
  assert(vs.size() == N + 1);
  assert(lbdas.size() == N + 1);
  for (uint i = 0; i <= N; i++) {
    const FactorData &d = datas[i];
    ConstVectorRef kff = d.ff[0];
    ConstVectorRef zff = d.ff[1];
    ConstVectorRef lff = d.ff[2];
    ConstVectorRef yff = d.ff[3];

    ConstMatrixRef K = d.fb.blockRow(0);
    ConstMatrixRef Z = d.fb.blockRow(1);
    ConstMatrixRef Lfb = d.fb.blockRow(2);
    ConstMatrixRef Yfb = d.fb.blockRow(3);

    ConstMatrixRef Kth = d.fth.blockRow(0);
    ConstMatrixRef Zth = d.fth.blockRow(1);
    ConstMatrixRef Lth = d.fth.blockRow(2);
    ConstMatrixRef Yth = d.fth.blockRow(3);

    us[i].noalias() = kff + K * xs[i];
    vs[i].noalias() = zff + Z * xs[i];
    if (theta_.has_value()) {
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
  ALIGATOR_NOMALLOC_END;
  return true;
}
} // namespace aligator::gar
