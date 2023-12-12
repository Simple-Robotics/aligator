/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
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
    VectorRef kff = d.ff.blockSegment(0);
    VectorRef zff = d.ff.blockSegment(1);
    RowMatrixRef K = d.fb.blockRow(0);
    RowMatrixRef Z = d.fb.blockRow(1);
    RowMatrixRef Kth = d.fth.blockRow(0);
    RowMatrixRef Zth = d.fth.blockRow(1);

    auto Ct = model.C.transpose();

    if (model.nu == 0) {
      Z = model.C / mueq;
      zff = model.d / mueq;
      Zth.setZero();
    } else {
      d.kktMat(0, 0) = model.R;
      d.kktMat(0, 1) = model.D.transpose();
      d.kktMat(1, 0) = model.D;
      d.kktMat(1, 1).diagonal().setConstant(-mueq);
      d.kktChol.compute(d.kktMat.data);

      kff = -model.r;
      zff = -model.d;
      K = -model.S.transpose();
      Z = -model.C;

      auto ffview = d.ff.template topBlkRows<2>();
      auto fbview = d.fb.template topBlkRows<2>();
      d.kktChol.solveInPlace(ffview.data);
      d.kktChol.solveInPlace(fbview.data);

      if (problem.isParameterized()) {
        Kth = -model.Gu;
        Zth.setZero();
        auto fthview = d.fth.template topBlkRows<2>();
        d.kktChol.solveInPlace(fthview.data);
      }
    }

    vc.Pmat.noalias() = model.Q + Ct * Z;
    vc.pvec.noalias() = model.q + Ct * zff;

    if (model.nu > 0) {
      vc.Pmat.noalias() += model.S * K;
      vc.pvec.noalias() += model.S * kff;
    }

    if (problem.isParameterized()) {
      vc.Lmat.noalias() = model.Gx + K.transpose() * model.Gu;
      vc.Psi.noalias() = model.Gth + model.Gu.transpose() * Kth;
      vc.svec.noalias() = model.gamma + model.B * kff;
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
    BlkMatrix<VectorRef, 2, 1> ffview = d.ff.template topBlkRows<2>();
    BlkMatrix<RowMatrixRef, 2, 1> fbview = d.fb.template topBlkRows<2>();
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

      d.Gxhat.noalias() = model.Gx + d.AtV * Xith;
      d.Guhat.noalias() = model.Gu + d.BtV * Xith;

      // set rhs of 2x2 block system and solve
      Kth = -d.Guhat;
      Zth.setZero();
      BlkMatrix<RowMatrixRef, 2, 1> fthview = d.fth.template topBlkRows<2>();
      d.kktChol.solveInPlace(fthview.data);

      // substitute into Xith, Ath gains
      Xith += model.B * Kth;
      vn.schurChol.solveInPlace(Xith);
      Ath += -d.PinvEt * Xith;

      // update s, L, Psi
      vc.svec = vn.svec + model.gamma;
      // vc.svec.noalias() += d.Guhat.transpose() * kff;
      vc.svec.noalias() += model.Gu.transpose() * kff;
      vc.svec.noalias() += vn.Lmat.transpose() * a;

      // vc.Lmat.noalias() = d.Gxhat + K.transpose() * d.Guhat;
      vc.Lmat = model.Gx;
      vc.Lmat.noalias() += K.transpose() * model.Gu;
      vc.Lmat.noalias() += A.transpose() * vn.Lmat;

      vc.Psi = model.Gth + vn.Psi;
      vc.Psi.noalias() += model.Gu.transpose() * Kth;
      vc.Psi.noalias() += vn.Lmat.transpose() * Ath;
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
  if (solveInitial) {
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
    thHess.noalias() =
        vinit.Psi + vinit.Lmat.transpose() * kkt0.fth.blockRow(0);
  }

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
  vnext.schurMat.noalias() = model.E * d.PinvEt;
  vnext.schurMat.diagonal().array() += mudyn;
  vnext.schurChol.compute(vnext.schurMat);
  // evaluate inverse of schurMat
  vnext.Vmat.setIdentity();
  vnext.schurChol.solveInPlace(vnext.Vmat);

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
    std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
    std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
    const boost::optional<ConstVectorRef> &theta_) const {
  ALIGATOR_NOMALLOC_BEGIN;
  // solve initial stage
  if (solveInitial) {
    computeInitial(xs[0], lbdas[0], theta_);
  }

  uint N = (uint)problem.horizon();
  for (uint t = 0; t <= N; t++) {
    const stage_factor_t &d = datas[t];
    const knot_t &model = problem.stages[t];
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

    if (problem.isParameterized() && theta_.has_value()) {
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

    if (problem.isParameterized() && theta_.has_value()) {
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

template <typename Scalar> void ProximalRiccatiSolver<Scalar>::initialize() {
  auto N = uint(problem.horizon());
  auto &knots = problem.stages;
  datas.reserve(N + 1);
  for (uint t = 0; t <= N; t++) {
    const knot_t &knot = knots[t];
    datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nth);
  }
  thGrad.setZero();
  thHess.setZero();
}

} // namespace gar
} // namespace aligator
