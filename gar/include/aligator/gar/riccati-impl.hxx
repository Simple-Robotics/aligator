/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./riccati-impl.hpp"
#include "./lqr-problem.hpp"

#include <tracy/Tracy.hpp>

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
  terminalSolve(stages[N], mueq, datas[N]);

  if (N == 0)
    return true;

  uint t = N - 1;
  while (true) {
    value_t &vn = datas[t + 1].vm;
    stageKernelSolve(stages[t], datas[t], vn, mudyn, mueq);

    if (t == 0)
      break;
    --t;
  }

  return true;
}
template <typename Scalar>
void ProximalRiccatiKernel<Scalar>::terminalSolve(const KnotType &model,
                                                  const Scalar mueq,
                                                  StageFactorType &d) {
  ZoneScoped;
  value_t &vc = d.vm;
  // fill cost-to-go matrix
  VectorRef kff = d.ff.blockSegment(0);
  VectorRef zff = d.ff.blockSegment(1);
  RowMatrixRef K = d.fb.blockRow(0);
  RowMatrixRef Z = d.fb.blockRow(1);
  RowMatrixRef Kth = d.fth.blockRow(0);
  RowMatrixRef Zth = d.fth.blockRow(1);

  Eigen::Transpose<const MatrixXs> Ct = model.C.transpose();

  if (model.nu == 0) {
    Z = model.C / mueq;
    zff = model.d / mueq;
    Zth.setZero();
  } else {
    d.kktMat(0, 0) = model.R;
    d.kktMat(0, 1) = model.D.transpose();
    d.kktMat(1, 0) = model.D;
    d.kktMat(1, 1).diagonal().setConstant(-mueq);
    d.kktChol.compute(d.kktMat.matrix());

    kff = -model.r;
    zff = -model.d;
    K = -model.S.transpose();
    Z = -model.C;

    auto ffview = d.ff.template topBlkRows<2>();
    auto fbview = d.fb.template topBlkRows<2>();
    d.kktChol.solveInPlace(ffview.matrix());
    d.kktChol.solveInPlace(fbview.matrix());

    if (model.nth > 0) {
      Kth = -model.Gu;
      Zth.setZero();
      auto fthview = d.fth.template topBlkRows<2>();
      d.kktChol.solveInPlace(fthview.matrix());
    }
  }

  vc.Pmat.noalias() = model.Q + Ct * Z;
  vc.pvec.noalias() = model.q + Ct * zff;

  if (model.nu > 0) {
    vc.Pmat.noalias() += model.S * K;
    vc.pvec.noalias() += model.S * kff;
  }

  if (model.nth > 0) {
    vc.Vxt = model.Gx;
    vc.Vxt.noalias() += K.transpose() * model.Gu;
    vc.Vtt = model.Gth;
    vc.Vtt.noalias() += model.Gu.transpose() * Kth;
    vc.vt = model.gamma;
    vc.vt.noalias() += model.Gu.transpose() * kff;
  }
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
                                                     value_t &vn,
                                                     const Scalar mudyn,
                                                     const Scalar mueq) {
  ZoneScoped;
  // step 1. compute decomposition of the E matrix
  d.Efact.compute(model.E);
  d.EinvP.setIdentity();
  d.Einv = d.Efact.solve(d.EinvP);
  auto &ptilde = vn.vx; // just an alias
  ptilde.noalias() = d.Einv.transpose() * vn.pvec;
  ptilde *= -1;
  d.EinvP.noalias() = d.Einv.transpose() * vn.Pmat;
  d.Ptilde.noalias() = d.EinvP * d.Einv;
  d.Ptilde = d.Ptilde.template selfadjointView<Eigen::Lower>();

  d.schurMat.setIdentity();
  d.schurMat.noalias() += mudyn * d.Ptilde;

  d.schurChol.compute(d.schurMat);
  vn.Vxx = d.Ptilde;
  vn.vx.noalias() += d.Ptilde * model.f;
  d.schurChol.solveInPlace(vn.vx);
  d.schurChol.solveInPlace(vn.Vxx);
  vn.Vxx = vn.Vxx.template selfadjointView<Eigen::Lower>();

  d.AtV.noalias() = model.A.transpose() * vn.Vxx;
  d.BtV.noalias() = model.B.transpose() * vn.Vxx;

  d.Qhat.noalias() = model.Q + d.AtV * model.A;
  d.Rhat.noalias() = model.R + d.BtV * model.B;
  d.Shat.noalias() = model.S + d.AtV * model.B;
  d.qhat.noalias() = model.q + model.A.transpose() * vn.vx;
  d.rhat.noalias() = model.r + model.B.transpose() * vn.vx;

  // factorize reduced KKT system
  d.kktMat(0, 0) = d.Rhat;
  d.kktMat(0, 1) = model.D.transpose();
  d.kktMat(1, 0) = model.D;
  d.kktMat(1, 1).diagonal().setConstant(-mueq);
  d.kktMat.matrix() =
      d.kktMat.matrix().template selfadjointView<Eigen::Lower>();
  d.kktChol.compute(d.kktMat.matrix());

  VectorRef kff = d.ff.blockSegment(0);
  VectorRef zff = d.ff.blockSegment(1);
  VectorRef lff = d.ff.blockSegment(2);
  VectorRef yff = d.ff.blockSegment(3);

  // fill feedback system
  kff = -d.rhat;
  zff = -model.d;

  RowMatrixRef K = d.fb.blockRow(0);
  RowMatrixRef Z = d.fb.blockRow(1);
  RowMatrixRef L = d.fb.blockRow(2);
  RowMatrixRef A = d.fb.blockRow(3);
  K = -d.Shat.transpose();
  Z = -model.C;
  auto ffview = d.ff.template topBlkRows<2>();
  auto fbview = d.fb.template topBlkRows<2>();
  d.kktChol.solveInPlace(ffview.matrix());
  d.kktChol.solveInPlace(fbview.matrix());

  // set closed loop dynamics
  lff.noalias() = vn.vx + d.BtV.transpose() * kff;
  yff.noalias() = model.f + model.B * kff;
  yff -= mudyn * lff;
  d.yff_pre = yff;
  yff.noalias() = d.Einv * d.yff_pre;
  yff *= -1;

  L.noalias() = vn.Vxx * model.A;
  L.noalias() += d.BtV.transpose() * K;

  A.noalias() = model.A + model.B * K;
  A -= mudyn * L;
  d.A_pre = A;
  A.noalias() = d.Einv * d.A_pre;
  A *= -1;

  value_t &vc = d.vm;
  Eigen::Transpose Ct = model.C.transpose();
  vc.Pmat.noalias() = d.Qhat + d.Shat * K + Ct * Z;
  vc.pvec.noalias() = d.qhat + d.Shat * kff + Ct * zff;

  RowMatrixRef Kth = d.fth.blockRow(0);
  RowMatrixRef Zth = d.fth.blockRow(1);
  RowMatrixRef Lth = d.fth.blockRow(2);
  RowMatrixRef Yth = d.fth.blockRow(3);
  if (model.nth > 0) {
    ZoneScopedN("stage_solve_parameter");

    // store Pxttilde = -Einv * Pxt
    // this is like ptilde
    Lth.noalias() = d.Einv.transpose() * vn.Vxt;
    Lth *= -1;
    auto &Pxttilde = Lth; // just an alias for clarity
    // store Lambda.inv * Pxttilde
    d.schurChol.solveInPlace(Lth);

    // d.Gxhat.noalias() = model.Gx + model.A.transpose() * Pxttilde;
    d.Guhat.noalias() = model.Gu + model.B.transpose() * Pxttilde;

    // set rhs of 2x2 block system and solve
    Kth = -d.Guhat;
    Zth.setZero();
    BlkMatrix<RowMatrixRef, 2, 1> fthview = d.fth.template topBlkRows<2>();
    d.kktChol.solveInPlace(fthview.matrix());

    // substitute into Xith, Ath gains
    Lth.noalias() += d.BtV.transpose() * Kth;

    Yth.noalias() = model.B * Kth;
    Yth -= mudyn * Lth;
    d.Yth_pre = Yth;
    Yth.noalias() = d.Einv * d.Yth_pre;
    Yth *= -1;

    // update vt, Vxt, Vtt
    vc.vt = vn.vt + model.gamma;
    // vc.vt.noalias() += d.Guhat.transpose() * kff;
    vc.vt.noalias() += model.Gu.transpose() * kff;
    vc.vt.noalias() += vn.Vxt.transpose() * yff;

    // vc.Vxt.noalias() = d.Gxhat + K.transpose() * d.Guhat;
    vc.Vxt = model.Gx;
    vc.Vxt.noalias() += K.transpose() * model.Gu;
    vc.Vxt.noalias() += A.transpose() * vn.Vxt;

    vc.Vtt = model.Gth + vn.Vtt;
    vc.Vtt.noalias() += model.Gu.transpose() * Kth;
    vc.Vtt.noalias() += vn.Vxt.transpose() * Yth;
  }
}

template <typename Scalar>
bool ProximalRiccatiKernel<Scalar>::forwardImpl(
    boost::span<const KnotType> stages,
    boost::span<const StageFactorType> datas, boost::span<VectorXs> xs,
    boost::span<VectorXs> us, boost::span<VectorXs> vs,
    boost::span<VectorXs> lbdas, const std::optional<ConstVectorRef> &theta_) {
  ZoneScoped;

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
  return true;
}

} // namespace gar
} // namespace aligator
