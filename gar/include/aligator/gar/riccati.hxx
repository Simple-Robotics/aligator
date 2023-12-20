/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./riccati.hpp"

namespace aligator {
namespace gar {
template <typename Scalar>
bool ProximalRiccatiSolver<Scalar>::backward(const Scalar mudyn,
                                             const Scalar mueq) {
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
    stage_factor_t &d = datas[t];
    value_t &vn = datas[t + 1].vm;
    const knot_t &model = problem.stages[t];
    solveOneStage(model, d, vn, mudyn, mueq);

    if (t == 0)
      break;
    --t;
  }

  stage_factor_t &d0 = datas[0];
  value_t &vinit = d0.vm;
  vinit.Vxx = vinit.Pmat;
  vinit.vx = vinit.pvec;

  // initial stage
  if (solveInitial) {
    kkt0.mat(0, 0) = vinit.Vxx;
    kkt0.mat(1, 0) = problem.G0;
    kkt0.mat(0, 1) = problem.G0.transpose();
    kkt0.mat(1, 1).diagonal().setConstant(-mudyn);
    kkt0.chol.compute(kkt0.mat.matrix());

    kkt0.ff.blockSegment(0) = -vinit.vx;
    kkt0.ff.blockSegment(1) = -problem.g0;
    kkt0.chol.solveInPlace(kkt0.ff.matrix());
    kkt0.fth.blockRow(0) = -vinit.Vxt;
    kkt0.fth.blockRow(1).setZero();
    kkt0.chol.solveInPlace(kkt0.fth.matrix());

    thGrad.noalias() =
        vinit.vt + vinit.Vxt.transpose() * kkt0.ff.blockSegment(0);
    thHess.noalias() = vinit.Vtt + vinit.Vxt.transpose() * kkt0.fth.blockRow(0);
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
