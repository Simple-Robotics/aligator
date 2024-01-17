/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./riccati-impl.hpp"

namespace aligator {
namespace gar {
template <typename Scalar>
bool ProximalRiccatiImpl<Scalar>::backwardImpl(
    boost::span<const KnotType> stages, const Scalar mudyn, const Scalar mueq,
    boost::span<StageFactor> datas) {
  // terminal node
  uint N = (uint)(datas.size() - 1);
  {
    StageFactor &d = datas[N];
    value_t &vc = d.vm;
    const KnotType &model = stages[N];
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
      vc.vt.noalias() += model.B * kff;
    }
  }

  uint t = N - 1;
  while (true) {
    value_t &vn = datas[t + 1].vm;
    solveOneStage(stages[t], datas[t], vn, mudyn, mueq);

    if (t == 0)
      break;
    --t;
  }

  return true;
}

template <typename Scalar>
void ProximalRiccatiImpl<Scalar>::computeMatrixTerms(const KnotType &model,
                                                     Scalar mudyn, Scalar mueq,
                                                     value_t &vnext,
                                                     StageFactor &d) {
  vnext.Pchol.compute(vnext.Pmat);
  d.PinvEt = vnext.Pchol.solve(model.E.transpose());
  vnext.schurMat.noalias() = model.E * d.PinvEt;
  vnext.schurMat.diagonal().array() += mudyn;
  vnext.schurChol.compute(vnext.schurMat);
  // evaluate inverse of schurMat
  vnext.Vxx.setIdentity();
  vnext.schurChol.solveInPlace(vnext.Vxx);

  d.AtV.noalias() = model.A.transpose() * vnext.Vxx;
  d.BtV.noalias() = model.B.transpose() * vnext.Vxx;

  d.Qhat.noalias() = model.Q + d.AtV * model.A;
  d.Rhat.noalias() = model.R + d.BtV * model.B;
  d.Shat.noalias() = model.S + d.AtV * model.B;

  // factorize reduced KKT system
  d.kktMat(0, 0) = d.Rhat;
  d.kktMat(0, 1) = model.D.transpose();
  d.kktMat(1, 0) = model.D;
  d.kktMat(1, 1).diagonal().setConstant(-mueq);
  d.kktChol.compute(d.kktMat.matrix());
}

template <typename Scalar>
void ProximalRiccatiImpl<Scalar>::solveOneStage(const KnotType &model,
                                                StageFactor &d, value_t &vn,
                                                const Scalar mudyn,
                                                const Scalar mueq) {

  // compute matrix expressions for the inverse
  computeMatrixTerms(model, mudyn, mueq, vn, d);

  VectorRef kff = d.ff.blockSegment(0);
  VectorRef zff = d.ff.blockSegment(1);
  VectorRef xi = d.ff.blockSegment(2);
  VectorRef a = d.ff.blockSegment(3);
  a = vn.Pchol.solve(-vn.pvec);

  vn.vx.noalias() = model.f + model.E * a;

  // fill feedback system
  d.qhat.noalias() = model.q + d.AtV * vn.vx;
  d.rhat.noalias() = model.r + d.BtV * vn.vx;
  kff = -d.rhat;
  zff = -model.d;

  RowMatrixRef K = d.fb.blockRow(0);
  RowMatrixRef Z = d.fb.blockRow(1);
  RowMatrixRef Xi = d.fb.blockRow(2);
  RowMatrixRef A = d.fb.blockRow(3);
  K = -d.Shat.transpose();
  Z = -model.C;
  BlkMatrix<VectorRef, 2, 1> ffview = d.ff.template topBlkRows<2>();
  BlkMatrix<RowMatrixRef, 2, 1> fbview = d.fb.template topBlkRows<2>();
  d.kktChol.solveInPlace(ffview.matrix());
  d.kktChol.solveInPlace(fbview.matrix());

  // set closed loop dynamics
  xi.noalias() = vn.Vxx * vn.vx;
  xi.noalias() += d.BtV.transpose() * kff;

  Xi.noalias() = vn.Vxx * model.A;
  Xi.noalias() += d.BtV.transpose() * K;

  a.noalias() -= d.PinvEt * xi;
  A.noalias() = -d.PinvEt * Xi;

  value_t &vc = d.vm;
  Eigen::Transpose<const MatrixXs> Ct = model.C.transpose();
  vc.Pmat.noalias() = d.Qhat + d.Shat * K + Ct * Z;
  vc.pvec.noalias() = d.qhat + d.Shat * kff + Ct * zff;

  if (model.nth > 0) {
    RowMatrixRef Kth = d.fth.blockRow(0);
    RowMatrixRef Zth = d.fth.blockRow(1);
    RowMatrixRef Xith = d.fth.blockRow(2);
    RowMatrixRef Ath = d.fth.blockRow(3);

    // store -Pinv * L
    Ath = vn.Pchol.solve(-vn.Vxt);
    // store -V * E * Pinv * L
    Xith.noalias() = model.E * Ath;

    d.Gxhat.noalias() = model.Gx + d.AtV * Xith;
    d.Guhat.noalias() = model.Gu + d.BtV * Xith;

    // set rhs of 2x2 block system and solve
    Kth = -d.Guhat;
    Zth.setZero();
    BlkMatrix<RowMatrixRef, 2, 1> fthview = d.fth.template topBlkRows<2>();
    d.kktChol.solveInPlace(fthview.matrix());

    // substitute into Xith, Ath gains
    Xith.noalias() += model.B * Kth;
    vn.schurChol.solveInPlace(Xith);
    Ath.noalias() -= d.PinvEt * Xith;

    // update vt, Vxt, Vtt
    vc.vt = vn.vt + model.gamma;
    // vc.vt.noalias() += d.Guhat.transpose() * kff;
    vc.vt.noalias() += model.Gu.transpose() * kff;
    vc.vt.noalias() += vn.Vxt.transpose() * a;

    // vc.Vxt.noalias() = d.Gxhat + K.transpose() * d.Guhat;
    vc.Vxt = model.Gx;
    vc.Vxt.noalias() += K.transpose() * model.Gu;
    vc.Vxt.noalias() += A.transpose() * vn.Vxt;

    vc.Vtt = model.Gth + vn.Vtt;
    vc.Vtt.noalias() += model.Gu.transpose() * Kth;
    vc.Vtt.noalias() += vn.Vxt.transpose() * Ath;
  }
}

template <typename Scalar>
bool ProximalRiccatiImpl<Scalar>::forwardImpl(
    boost::span<const KnotType> stages, boost::span<const StageFactor> datas,
    boost::span<VectorXs> xs, boost::span<VectorXs> us,
    boost::span<VectorXs> vs, boost::span<VectorXs> lbdas,
    const std::optional<ConstVectorRef> &theta_) {
  ALIGATOR_NOMALLOC_BEGIN;

  uint N = (uint)(datas.size() - 1);
  for (uint t = 0; t <= N; t++) {
    const StageFactor &d = datas[t];
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
