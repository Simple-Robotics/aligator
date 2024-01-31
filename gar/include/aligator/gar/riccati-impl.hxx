/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./riccati-impl.hpp"
#include "tracy/Tracy.hpp"

namespace aligator {
namespace gar {
template <typename Scalar>
bool ProximalRiccatiKernel<Scalar>::backwardImpl(
    boost::span<const KnotType> stages, const Scalar mudyn, const Scalar mueq,
    boost::span<StageFactorType> datas) {
  ZoneScoped;
  // terminal node
  if (datas.size() == 0)
    return true;
  uint N = (uint)(datas.size() - 1);
  stageKernelSolve(stages[N], datas[N], nullptr, mudyn, mueq);
  if (N == 0)
    return true;

  uint t = N - 1;
  while (true) {
    value_t &vn = datas[t + 1].vm;
    stageKernelSolve(stages[t], datas[t], &vn, mudyn, mueq);

    if (t == 0)
      break;
    --t;
  }

  return true;
}

template <typename Scalar>
void ProximalRiccatiKernel<Scalar>::computeInitial(
    VectorRef x0, VectorRef lbd0, const kkt0_t &kkt0,
    const std::optional<ConstVectorRef> &theta_) {
  ZoneScoped;
  assert(kkt0.chol.info() == Eigen::Success);
  x0 = kkt0.ff.blockSegment(0);
  lbd0 = kkt0.ff.blockSegment(1);
  if (theta_.has_value()) {
    x0.noalias() += kkt0.fth.blockRow(0) * theta_.value();
    lbd0.noalias() += kkt0.fth.blockRow(1) * theta_.value();
  }
}

template <typename Scalar>
void ProximalRiccatiKernel<Scalar>::stageKernelSolve(const KnotType &model,
                                                     StageFactorType &d,
                                                     value_t *vn,
                                                     const Scalar mudyn,
                                                     const Scalar mueq) {
  ZoneScoped;
  value_t &vc = d.vm;

  VectorRef kff = d.ff.blockSegment(0);
  VectorRef zff = d.ff.blockSegment(1);
  VectorRef lff = d.ff.blockSegment(2);
  VectorRef yff = d.ff.blockSegment(3);
  RowMatrixRef K = d.fb.blockRow(0);
  RowMatrixRef Z = d.fb.blockRow(1);
  RowMatrixRef L = d.fb.blockRow(2);
  RowMatrixRef Y = d.fb.blockRow(3);
  RowMatrixRef Kth = d.fth.blockRow(0);
  RowMatrixRef Zth = d.fth.blockRow(1);
  RowMatrixRef Lth = d.fth.blockRow(2);
  RowMatrixRef Yth = d.fth.blockRow(3);

  // terminal
  if (model.nx2 == 0)
    vn = nullptr;

  d.Qhat = model.Q;
  d.Rhat = model.R;
  d.Shat = model.S;
  d.qhat = model.q;
  d.rhat = model.r;

  if (vn) {
    vn->Pchol.compute(vn->Pmat);
    d.PinvEt = vn->Pchol.solve(model.E.transpose());
    vn->schurMat.noalias() = model.E * d.PinvEt;
    vn->schurMat.diagonal().array() += mudyn;
    vn->schurChol.compute(vn->schurMat);
    // evaluate inverse of schurMat
    vn->Vxx.setIdentity();
    vn->schurChol.solveInPlace(vn->Vxx);

    d.AtV.noalias() = model.A.transpose() * vn->Vxx;
    d.BtV.noalias() = model.B.transpose() * vn->Vxx;

    d.Qhat.noalias() += d.AtV * model.A;
    d.Rhat.noalias() += d.BtV * model.B;
    d.Shat.noalias() += d.AtV * model.B;

    yff = vn->Pchol.solve(-vn->pvec);
    vn->vx.noalias() = model.f + model.E * yff;

    d.qhat.noalias() += d.AtV * vn->vx;
    d.rhat.noalias() += d.BtV * vn->vx;
  }

  // factorize reduced KKT system
  d.kktMat(0, 0) = d.Rhat;
  d.kktMat(0, 1) = model.D.transpose();
  d.kktMat(1, 0) = model.D;
  d.kktMat(1, 1).diagonal().setConstant(-mueq);
  d.kktChol.compute(d.kktMat.matrix());

  kff = -d.rhat;
  zff = -model.d;
  K = -d.Shat.transpose();
  Z = -model.C;
  BlkMatrix ffview = d.ff.template topBlkRows<2>();
  BlkMatrix fbview = d.fb.template topBlkRows<2>();
  d.kktChol.solveInPlace(ffview.matrix());
  d.kktChol.solveInPlace(fbview.matrix());

  // set closed loop dynamics
  if (vn) {
    lff.noalias() = vn->Vxx * vn->vx;
    lff.noalias() += d.BtV.transpose() * kff;

    L.noalias() = vn->Vxx * model.A;
    L.noalias() += d.BtV.transpose() * K;

    yff.noalias() -= d.PinvEt * lff;
    Y.noalias() = d.PinvEt * L;
    Y *= -1;
  }

  Eigen::Transpose<const MatrixXs> Ct = model.C.transpose();
  vc.Pmat.noalias() = d.Qhat + d.Shat * K;
  vc.Pmat.noalias() += Ct * Z;
  vc.pvec.noalias() = d.qhat + d.Shat * kff;
  vc.pvec.noalias() += Ct * zff;

  if (model.nth > 0) {
    ZoneScopedN("stage_solve_parameter");

    d.Gxhat = model.Gx;
    d.Guhat = model.Gu;
    if (vn) {
      // store -Pinv * L
      Yth = vn->Pchol.solve(-vn->Vxt);
      // store -V * E * Pinv * L
      Lth.noalias() = model.E * Yth;
      d.Gxhat.noalias() += d.AtV * Lth;
      d.Guhat.noalias() += d.BtV * Lth;
    }

    // set rhs of 2x2 block system and solve
    Kth = -d.Guhat;
    Zth.setZero();
    BlkMatrix<RowMatrixRef, 2, 1> fthview = d.fth.template topBlkRows<2>();
    // solve for (u,v)-gains
    d.kktChol.solveInPlace(fthview.matrix());

    // update vt, Vxt, Vtt
    vc.vt = model.gamma;
    // vc.vt.noalias() += d.Guhat.transpose() * kff;
    vc.vt.noalias() += model.Gu.transpose() * kff;
    vc.vt.noalias() += model.Gv.transpose() * zff;

    // vc.Vxt.noalias() = d.Gxhat + K.transpose() * d.Guhat;
    vc.Vxt = d.Gxhat;
    vc.Vxt.noalias() += d.Shat * Kth;
    vc.Vxt.noalias() += Ct * Zth;

    vc.Vtt = model.Gth;
    vc.Vtt.noalias() += model.Gu.transpose() * Kth;
    vc.Vtt.noalias() += model.Gv.transpose() * Zth;
    if (vn) {
      vc.vt += vn->vt;
      vc.vt.noalias() += vn->Vxt.transpose() * yff;
      vc.Vtt += vn->Vtt;
      // forward-substitute into (lbda,xtp1) param gains
      Lth.noalias() += model.B * Kth;
      vn->schurChol.solveInPlace(Lth);
      Yth.noalias() -= d.PinvEt * Lth;
      vc.Vtt.noalias() += vn->Vxt.transpose() * Yth;
    }
  }
}

template <typename Scalar>
bool ProximalRiccatiKernel<Scalar>::forwardImpl(
    boost::span<const KnotType> stages,
    boost::span<const StageFactorType> datas, boost::span<VectorXs> xs,
    boost::span<VectorXs> us, boost::span<VectorXs> vs,
    boost::span<VectorXs> lbdas, const std::optional<ConstVectorRef> &theta_) {
  ZoneScoped;
  ALIGATOR_NOMALLOC_BEGIN;

  uint N = (uint)(datas.size() - 1);
  for (uint t = 0; t <= N; t++) {
    const StageFactorType &d = datas[t];
    const KnotType &model = stages[t];
    assert(xs[t].size() == model.nx);
    assert(vs[t].size() == model.nc);

    ConstRowMatrixRef K = d.fb.blockRow(0); // control feedback
    ConstVectorRef kff = d.ff.blockSegment(0);
    if (model.nu > 0) {
      assert(us[t].size() == model.nu);
      us[t].noalias() = kff + K * xs[t];
    }

    ConstRowMatrixRef Z = d.fb.blockRow(1); // multiplier feedback
    ConstVectorRef zff = d.ff.blockSegment(1);
    vs[t].noalias() = zff + Z * xs[t];

    if (model.nth > 0 && theta_.has_value()) {
      ConstVectorRef theta = *theta_;
      ConstRowMatrixRef Kth = d.fth.blockRow(0);
      ConstRowMatrixRef Zth = d.fth.blockRow(1);

      if (model.nu > 0)
        us[t].noalias() += Kth * theta;
      vs[t].noalias() += Zth * theta;
    }

    if (t == N)
      break;

    assert(lbdas[t + 1].size() == model.f.size());

    ConstRowMatrixRef Xi = d.fb.blockRow(2);
    ConstVectorRef xi = d.ff.blockSegment(2);
    lbdas[t + 1].noalias() = xi + Xi * xs[t];

    ConstRowMatrixRef A = d.fb.blockRow(3);
    ConstVectorRef a = d.ff.blockSegment(3);
    xs[t + 1].noalias() = a + A * xs[t];

    if (model.nth > 0 && theta_.has_value()) {
      ConstVectorRef theta = *theta_;
      ConstRowMatrixRef Xith = d.fth.blockRow(2);
      ConstRowMatrixRef Ath = d.fth.blockRow(3);

      lbdas[t + 1].noalias() += Xith * theta;
      xs[t + 1].noalias() += Ath * theta;
    }
  }

  ALIGATOR_NOMALLOC_END;
  return true;
}

} // namespace gar
} // namespace aligator
