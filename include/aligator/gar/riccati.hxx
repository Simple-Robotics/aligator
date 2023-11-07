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
  }

  uint t = N - 1;
  while (true) {
    stage_factor_t &d = datas[t];
    value_t &vn = datas[t + 1].vm;
    const knot_t &model = problem.stages[t];

    vn.Pchol.compute(vn.Pmat);
    d.PinvEt = vn.Pchol.solve(model.E.transpose());
    d.Pinvp = vn.Pchol.solve(vn.pvec);

    ALIGATOR_NOMALLOC_END;
    {
      d.err.pm = math::infty_norm(vn.Pmat * d.PinvEt - model.E.transpose());
      d.err.pv = math::infty_norm(vn.Pmat * d.Pinvp + vn.pvec);
    }
    ALIGATOR_NOMALLOC_BEGIN;

    vn.Lbmat.noalias() = model.E * d.PinvEt;
    vn.Lbmat.diagonal().array() += mudyn;

    vn.Lbchol.compute(vn.Lbmat);
    vn.Vmat.setIdentity();
    vn.Lbchol.solveInPlace(vn.Vmat); // evaluate inverse of Lambda
    vn.vvec.noalias() = model.f - model.E * d.Pinvp;
    ALIGATOR_NOMALLOC_END;
    d.err.lbda = math::infty_norm(vn.Lbmat - vn.Lbchol.reconstructedMatrix());
    ALIGATOR_NOMALLOC_BEGIN;

    // fill in hamiltonian
    computeKktTerms(model, d, vn);

    // fill feedback system
    d.kktMat(0, 0) = d.hmlt.Rhat;
    d.kktMat(1, 0) = model.D;
    d.kktMat(1, 1).diagonal().setConstant(-mueq);
    d.kktMat.data = d.kktMat.data.template selfadjointView<Eigen::Lower>();
    d.kktChol.compute(d.kktMat.data);

    VectorRef kff = d.ff.blockSegment(0);
    VectorRef zff = d.ff.blockSegment(1);
    kff = -d.hmlt.rhat;
    zff = -model.d;

    RowMatrixRef K = d.fb.blockRow(0);
    RowMatrixRef Z = d.fb.blockRow(1);
    K = -d.hmlt.Shat.transpose();
    Z = -model.C;
    BlkMatrix<VectorRef, 2, 1> ffview = topBlkRows<2>(d.ff);
    BlkMatrix<RowMatrixRef, 2, 1> fbview = topBlkRows<2>(d.fb);
    d.kktChol.solveInPlace(ffview.data);
    d.kktChol.solveInPlace(fbview.data);

    // set closed loop dynamics
    {
      auto xi = d.ff.blockSegment(2);
      auto a = d.ff.blockSegment(3);
      auto Xi = d.fb.blockRow(2);
      auto A = d.fb.blockRow(3);

      VectorRef tmpxivec = d.tmpClp.col(0);
      tmpxivec.noalias() = vn.vvec + model.B * kff;
      xi.noalias() = vn.Vmat * tmpxivec;

      MatrixRef tmpximat = d.tmpClp;
      tmpximat.noalias() = model.A + model.B * K;
      Xi.noalias() = vn.Vmat * tmpximat;

      a.noalias() = -d.Pinvp - d.PinvEt * xi;
      A.noalias() = -d.PinvEt * Xi;
    }

    value_t &vc = d.vm;
    auto Ct = model.C.transpose();
    vc.Pmat.noalias() = d.hmlt.Qhat + d.hmlt.Shat * K + Ct * Z;
    vc.pvec.noalias() = d.hmlt.qhat + d.hmlt.Shat * kff + Ct * zff;

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
  kkt0.rhs.blockSegment(0) = -vinit.vvec;
  kkt0.rhs.blockSegment(1) = -problem.g0;

  kkt0.chol.compute(kkt0.mat.data);
  kkt0.chol.solveInPlace(kkt0.rhs.data);

  ALIGATOR_NOMALLOC_END;

  return true;
}

template <typename Scalar>
void ProximalRiccatiSolver<Scalar>::computeKktTerms(const knot_t &model,
                                                    stage_factor_t &d,
                                                    const value_t &vnext) {
  hmlt_t &hmlt = d.hmlt;
  hmlt.AtV.noalias() = model.A.transpose() * vnext.Vmat;
  hmlt.BtV.noalias() = model.B.transpose() * vnext.Vmat;

  hmlt.Qhat.noalias() = model.Q + hmlt.AtV * model.A;
  hmlt.Rhat.noalias() = model.R + hmlt.BtV * model.B;
  hmlt.Shat.noalias() = model.S + hmlt.AtV * model.B;

  hmlt.qhat.noalias() = model.q + hmlt.AtV * vnext.vvec;
  hmlt.rhat.noalias() = model.r + hmlt.BtV * vnext.vvec;
}

template <typename Scalar>
bool ProximalRiccatiSolver<Scalar>::forward(vecvec_t &xs, vecvec_t &us,
                                            vecvec_t &vs,
                                            vecvec_t &lbdas) const {
  ALIGATOR_NOMALLOC_BEGIN;
  // solve initial stage
  {
    xs[0] = kkt0.rhs.blockSegment(0);
    lbdas[0] = kkt0.rhs.blockSegment(1);
  }

  uint N = (uint)problem.horizon();
  for (uint t = 0; t <= N; t++) {
    const stage_factor_t &d = datas[t];

    Eigen::Ref<const RowMatrixXs> Z = d.fb.blockRow(1); // multiplier feedback
    ConstVectorRef zff = d.ff.blockSegment(1);
    vs[t].noalias() = zff + Z * xs[t];

    if (t == N)
      break;

    Eigen::Ref<const RowMatrixXs> K = d.fb.blockRow(0); // control feedback
    ConstVectorRef kff = d.ff.blockSegment(0);
    us[t].noalias() = kff + K * xs[t];

    ConstVectorRef xi = d.ff.blockSegment(2);
    Eigen::Ref<const RowMatrixXs> Xi = d.fb.blockRow(2);
    lbdas[t + 1].noalias() = xi + Xi * xs[t];

    ConstVectorRef a = d.ff.blockSegment(3);
    Eigen::Ref<const RowMatrixXs> A = d.fb.blockRow(3);
    xs[t + 1].noalias() = a + A * xs[t];
  }

  ALIGATOR_NOMALLOC_END;
  return true;
}

} // namespace gar
} // namespace aligator
