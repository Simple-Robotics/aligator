#pragma once

#include "./riccati.hpp"

namespace aligator {
namespace gar {
template <typename Scalar>
bool ProximalRiccatiSolver<Scalar>::backward(Scalar mudyn, Scalar mueq) {
  if (!problem.isInitialized())
    return false;

  ALIGATOR_NOMALLOC_BEGIN;
  // terminal node
  uint N = (uint)problem.horizon();
  {
    stage_factor_t &d = datas[N];
    value_t &vc = d.vm;
    const knot_t &model = problem.stages[N];
    // fill cost-to-go matrix
    VectorRef zff = d.ff.blockSegment(1);
    RowMatrixRef Z = d.fb.blockRow(1);

    auto Ct = model.C.transpose();

    Z.noalias() = model.C / mueq;
    zff.noalias() = model.d / mueq;

    vc.Pmat.noalias() = model.Q + Ct * Z;
    vc.pvec.noalias() = model.q + Ct * zff;

    if (problem.isParameterized()) {
      vc.Lmat = model.Gammax;
      vc.Psi = model.Gammath;
      vc.svec = model.gamma;
    }
  }

  uint t = N - 1;
  while (true) {
    stage_factor_t &d = datas[t];
    value_t &vn = datas[t + 1].vm;
    const knot_t &model = problem.stages[t];

    // compute matrix expressions for the inverse
    computeMatrixTerms(model, mudyn, mueq, vn, d);

    VectorRef kff = d.ff.blockSegment(0);
    VectorRef zff = d.ff.blockSegment(1);
    VectorRef xi = d.ff.blockSegment(2);
    VectorRef a = d.ff.blockSegment(3);
    a = vn.Pchol.solve(-vn.pvec);

    vn.vvec.noalias() = model.f + model.E * a;

    // fill feedback system
    d.qhat.noalias() = model.q + d.AtV * vn.vvec;
    d.rhat.noalias() = model.r + d.BtV * vn.vvec;
    kff = -d.rhat;
    zff = -model.d;

    RowMatrixRef K = d.fb.blockRow(0);
    RowMatrixRef Z = d.fb.blockRow(1);
    RowMatrixRef Xi = d.fb.blockRow(2);
    RowMatrixRef A = d.fb.blockRow(3);
    K = -d.Shat.transpose();
    Z = -model.C;
    BlkMatrix<VectorRef, 2, 1> ffview = topBlkRows<2>(d.ff);
    BlkMatrix<RowMatrixRef, 2, 1> fbview = topBlkRows<2>(d.fb);
    d.kktChol.solveInPlace(ffview.data);
    d.kktChol.solveInPlace(fbview.data);

    // set closed loop dynamics
    {
      xi.noalias() = vn.Vmat * vn.vvec;
      xi.noalias() += d.BtV.transpose() * kff;

      Xi.noalias() = vn.Vmat * model.A;
      Xi.noalias() += d.BtV.transpose() * K;

      a.noalias() += -d.PinvEt * xi;
      A.noalias() = -d.PinvEt * Xi;
    }

    value_t &vc = d.vm;
    auto Ct = model.C.transpose();
    vc.Pmat.noalias() = d.Qhat + d.Shat * K + Ct * Z;
    vc.pvec.noalias() = d.qhat + d.Shat * kff + Ct * zff;

    if (problem.isParameterized()) {
      RowMatrixRef Kth = d.fth.blockRow(0);
      RowMatrixRef Zth = d.fth.blockRow(1);
      RowMatrixRef Xith = d.fth.blockRow(2);
      RowMatrixRef Ath = d.fth.blockRow(3);

      // store -Pinv * L
      Ath = vn.Pchol.solve(-vn.Lmat);
      // store -V * E * Pinv * L
      Xith.noalias() = model.E * Ath;

      d.Gxhat.noalias() = model.Gammax + d.AtV * Xith;
      d.Guhat.noalias() = model.Gammau + d.BtV * Xith;

      // set rhs of 2x2 block system and solve
      Kth = -d.Guhat;
      Zth.setZero();
      BlkMatrix<RowMatrixRef, 2, 1> fthview = topBlkRows<2>(d.fth);
      d.kktChol.solveInPlace(fthview.data);

      // substitute into Xith, Ath gains
      Xith += model.B * Kth;
      vn.Lbchol.solveInPlace(Xith);
      Ath += -d.PinvEt * Xith;

      // update s, L, Psi
      vc.svec = vn.svec + model.gamma;
      vc.svec.noalias() += d.Guhat.transpose() * kff;
      vc.Lmat.noalias() = d.Gxhat + K.transpose() * d.Guhat;
      vc.Psi = model.Gammath + vn.Psi;
      vc.Psi.noalias() += d.Guhat.transpose() * Kth;
    }

    if (t == 0)
      break;
    --t;
  }

  stage_factor_t &d0 = datas[0];
  value_t &vinit = d0.vm;
  vinit.Vmat = vinit.Pmat;
  vinit.vvec = vinit.pvec;

  // initial stage
  kkt0.mat(0, 0) = vinit.Vmat;
  kkt0.mat(1, 0) = problem.G0;
  kkt0.mat(0, 1) = problem.G0.transpose();
  kkt0.mat(1, 1).diagonal().setConstant(-mudyn);
  kkt0.chol.compute(kkt0.mat.data);

  kkt0.ff.blockSegment(0) = -vinit.vvec;
  kkt0.ff.blockSegment(1) = -problem.g0;
  kkt0.chol.solveInPlace(kkt0.ff.data);
  kkt0.fth.blockRow(0) = -vinit.Lmat;
  kkt0.fth.blockRow(1).setZero();
  kkt0.chol.solveInPlace(kkt0.fth.data);

  thGrad.noalias() =
      vinit.svec + vinit.Lmat.transpose() * kkt0.ff.blockSegment(0);
  thHess.noalias() = vinit.Psi + vinit.Lmat.transpose() * kkt0.fth.blockRow(0);

  ALIGATOR_NOMALLOC_END;

  return true;
}

template <typename Scalar>
void ProximalRiccatiSolver<Scalar>::computeMatrixTerms(const knot_t &model,
                                                       Scalar mudyn,
                                                       Scalar mueq,
                                                       value_t &vnext,
                                                       stage_factor_t &d) {
  vnext.Pchol.compute(vnext.Pmat);
  d.PinvEt = vnext.Pchol.solve(model.E.transpose());
  vnext.Lbmat.noalias() = model.E * d.PinvEt;
  vnext.Lbmat.diagonal().array() += mudyn;
  vnext.Lbchol.compute(vnext.Lbmat);
  // evaluate inverse of Lbmat
  vnext.Vmat.setIdentity();
  vnext.Lbchol.solveInPlace(vnext.Vmat);

  d.AtV.noalias() = model.A.transpose() * vnext.Vmat;
  d.BtV.noalias() = model.B.transpose() * vnext.Vmat;

  d.Qhat.noalias() = model.Q + d.AtV * model.A;
  d.Rhat.noalias() = model.R + d.BtV * model.B;
  d.Shat.noalias() = model.S + d.AtV * model.B;

  // factorize reduced KKT system
  d.kktMat(0, 0) = d.Rhat;
  d.kktMat(0, 1) = model.D.transpose();
  d.kktMat(1, 0) = model.D;
  d.kktMat(1, 1).diagonal().setConstant(-mueq);
  d.kktChol.compute(d.kktMat.data);
}

template <typename Scalar>
bool ProximalRiccatiSolver<Scalar>::forward(
    vecvec_t &xs, vecvec_t &us, vecvec_t &vs, vecvec_t &lbdas,
    const boost::optional<ConstVectorRef> &theta_) const {
  ALIGATOR_NOMALLOC_BEGIN;
  // solve initial stage
  {
    xs[0] = kkt0.ff.blockSegment(0);
    lbdas[0] = kkt0.ff.blockSegment(1);
    if (theta_.has_value()) {
      xs[0].noalias() += kkt0.fth.blockRow(0) * theta_.value();
      lbdas[0].noalias() += kkt0.fth.blockRow(1) * theta_.value();
    }
  }

  uint N = (uint)problem.horizon();
  for (uint t = 0; t <= N; t++) {
    const stage_factor_t &d = datas[t];

    ConstRowMatrixRef Z = d.fb.blockRow(1); // multiplier feedback
    ConstVectorRef zff = d.ff.blockSegment(1);
    vs[t].noalias() = zff + Z * xs[t];

    if (t == N)
      break;

    ConstRowMatrixRef K = d.fb.blockRow(0); // control feedback
    ConstVectorRef kff = d.ff.blockSegment(0);
    us[t].noalias() = kff + K * xs[t];

    ConstRowMatrixRef Xi = d.fb.blockRow(2);
    ConstVectorRef xi = d.ff.blockSegment(2);
    lbdas[t + 1].noalias() = xi + Xi * xs[t];

    ConstRowMatrixRef A = d.fb.blockRow(3);
    ConstVectorRef a = d.ff.blockSegment(3);
    xs[t + 1].noalias() = a + A * xs[t];

    if (problem.isParameterized() && theta_.has_value()) {
      ConstVectorRef theta = *theta_;
      assert(theta.rows() == problem.stages[0].nth);
      ConstRowMatrixRef Kth = d.fth.blockRow(0);
      ConstRowMatrixRef Zth = d.fth.blockRow(1);
      ConstRowMatrixRef Xith = d.fth.blockRow(2);
      ConstRowMatrixRef Ath = d.fth.blockRow(3);

      us[t].noalias() += Kth * theta;
      vs[t].noalias() += Zth * theta;
      lbdas[t + 1].noalias() += Xith * theta;
      xs[t + 1].noalias() += Ath * theta;
    }
  }

  ALIGATOR_NOMALLOC_END;
  return true;
}

} // namespace gar
} // namespace aligator
