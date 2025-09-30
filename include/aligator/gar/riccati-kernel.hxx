/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "riccati-kernel.hpp"

#include "aligator/tracy.hpp"

namespace aligator {
namespace gar {

template <typename Scalar>
StageFactor<Scalar>::StageFactor(uint nx, uint nu, uint nc, uint nx2, uint nth,
                                 const allocator_type &alloc)
    : nx(nx)
    , nu(nu)
    , nc(nc)
    , nx2(nx2)
    , nth(nth)
    , Qhat(nx, nx, alloc)
    , Rhat(nu, nu, alloc)
    , Shat(nx, nu, alloc)
    , qhat(nx, alloc)
    , rhat(nu, alloc)
    , AtV(nx, nx2, alloc)
    , BtV(nu, nx2, alloc)
    , Gxhat(nx, nth, alloc)
    , Guhat(nu, nth, alloc)
    , ff({nu, nc, nx2}, {1})
    , fb({nu, nc, nx2}, {nx})
    , fth({nu, nc, nx2}, {nth})
    , kktMat({nu, nc}, {nu, nc})
    , kktChol(nu + nc)
    , vm(nx, nth, alloc) {
  Qhat.setZero();
  Rhat.setZero();
  Shat.setZero();
  qhat.setZero();
  rhat.setZero();

  AtV.setZero();
  BtV.setZero();

  Gxhat.setZero();
  Guhat.setZero();

  ff.setZero();
  fb.setZero();
  fth.setZero();
  kktMat.setZero();
}

#define _c(name) name(other.name, alloc)
template <typename Scalar>
StageFactor<Scalar>::StageFactor(const StageFactor &other,
                                 const allocator_type &alloc)
    : nx(other.nx)
    , nu(other.nu)
    , nc(other.nc)
    , nx2(other.nx2)
    , nth(other.nth)
    , _c(Qhat)
    , _c(Rhat)
    , _c(Shat)
    , _c(qhat)
    , _c(rhat)
    , _c(AtV)
    , _c(BtV)
    , _c(Gxhat)
    , _c(Guhat)
    , ff(other.ff)
    , fb(other.fb)
    , fth(other.fth)
    , kktMat(other.kktMat)
    , kktChol(other.kktChol)
    , _c(vm) {}
#undef _c

#define _c(name) name(std::move(other.name), alloc)
template <typename Scalar>
StageFactor<Scalar>::StageFactor(StageFactor &&other,
                                 const allocator_type &alloc)
    : nx(other.nx)
    , nu(other.nu)
    , nc(other.nc)
    , nx2(other.nx2)
    , nth(other.nth)
    , _c(Qhat)
    , _c(Rhat)
    , _c(Shat)
    , _c(qhat)
    , _c(rhat)
    , _c(AtV)
    , _c(BtV)
    , _c(Gxhat)
    , _c(Guhat)
    , ff(std::move(other.ff))
    , fb(std::move(other.fb))
    , fth(std::move(other.fth))
    , kktMat(std::move(other.kktMat))
    , kktChol(std::move(other.kktChol))
    , _c(vm) {}
#undef _c

template <typename Scalar>
bool ProximalRiccatiKernel<Scalar>::backwardImpl(
    boost::span<const KnotType> stages, const Scalar mueq,
    boost::span<StageFactorType> datas) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  // terminal node
  if (datas.size() == 0)
    return true;
  uint N = (uint)(datas.size() - 1);
  terminalSolve(stages[N], mueq, datas[N]);

  if (N == 0)
    return true;

  uint t = N - 1;
  while (true) {
    CostToGo &vn = datas[t + 1].vm;
    stageKernelSolve(stages[t], datas[t], vn, mueq);

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
  ALIGATOR_TRACY_ZONE_SCOPED;
  CostToGo &vc = d.vm;
  // fill cost-to-go matrix
  VectorRef kff = d.ff.blockSegment(0);
  VectorRef zff = d.ff.blockSegment(1);
  RowMatrixRef K = d.fb.blockRow(0);
  RowMatrixRef Z = d.fb.blockRow(1);
  RowMatrixRef Kth = d.fth.blockRow(0);
  RowMatrixRef Zth = d.fth.blockRow(1);

  Eigen::Transpose Ct = model.C.transpose();

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

  vc.Vxx.noalias() = model.Q + Ct * Z;
  vc.vx.noalias() = model.q + Ct * zff;

  if (model.nu > 0) {
    vc.Vxx.noalias() += model.S * K;
    vc.vx.noalias() += model.S * kff;
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
  ALIGATOR_TRACY_ZONE_SCOPED;
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
                                                     CostToGo &vn,
                                                     const Scalar mueq) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  polymorphic_allocator alloc = d.get_allocator();
  vn.Vxx = vn.Vxx.template selfadjointView<Eigen::Lower>();
  ArenaMatrix<VectorXs> vplus{vn.vx, alloc};
  vplus += vn.Vxx * model.f;

  d.AtV.noalias() = model.A.transpose() * vn.Vxx;
  d.BtV.noalias() = model.B.transpose() * vn.Vxx;

  d.Qhat.noalias() = model.Q + d.AtV * model.A;
  d.Rhat.noalias() = model.R + d.BtV * model.B;
  d.Shat.noalias() = model.S + d.AtV * model.B;
  d.qhat.noalias() = model.q + model.A.transpose() * vplus;
  d.rhat.noalias() = model.r + model.B.transpose() * vplus;

  // factorize reduced KKT system
  d.kktMat(0, 0) = d.Rhat;
  d.kktMat(0, 1) = model.D.transpose();
  d.kktMat(1, 0) = model.D;
  d.kktMat(1, 1).diagonal().setConstant(-mueq);
  d.kktMat.matrix() =
      d.kktMat.matrix().template selfadjointView<Eigen::Lower>();
  d.kktChol.compute(d.kktMat.matrix());
  if (d.kktChol.info() != Eigen::Success) {
    ALIGATOR_RUNTIME_ERROR("Failed stage LDL factorization");
  }

  VectorRef kff = d.ff.blockSegment(0);
  VectorRef zff = d.ff.blockSegment(1);
  VectorRef yff = d.ff.blockSegment(2); // closed-loop bias

  // fill feedback system
  kff = -d.rhat;
  zff = -model.d;

  // rhs (feedback)
  RowMatrixRef K = d.fb.blockRow(0);
  RowMatrixRef Z = d.fb.blockRow(1);
  RowMatrixRef A = d.fb.blockRow(2); // closed-loop matrix
  K = -d.Shat.transpose();
  Z = -model.C;

  // solve
  auto ffview = d.ff.template topBlkRows<2>();
  auto fbview = d.fb.template topBlkRows<2>();
  d.kktChol.solveInPlace(ffview.matrix());
  d.kktChol.solveInPlace(fbview.matrix());

  // set closed loop dynamics
  yff.noalias() = model.f + model.B * kff;
  A.noalias() = model.A + model.B * K;

  CostToGo &vc = d.vm;
  Eigen::Transpose Ct = model.C.transpose();
  vc.Vxx.noalias() = d.Qhat + d.Shat * K + Ct * Z;
  vc.vx.noalias() = d.qhat + d.Shat * kff + Ct * zff;
  if (model.nth > 0) {
    ALIGATOR_TRACY_ZONE_SCOPED_N("stage_solve_parameter");
    RowMatrixRef Kth = d.fth.blockRow(0);
    RowMatrixRef Zth = d.fth.blockRow(1);
    RowMatrixRef Yth = d.fth.blockRow(2);

    d.Gxhat.noalias() = model.Gx + model.A.transpose() * vn.Vxt;
    d.Guhat.noalias() = model.Gu + model.B.transpose() * vn.Vxt;

    // set rhs of 2x2 block system and solve
    Kth = -d.Guhat;
    Zth.setZero();
    BlkMatrix<RowMatrixRef, 2, 1> fthview = d.fth.template topBlkRows<2>();
    d.kktChol.solveInPlace(fthview.matrix());

    Yth.noalias() = model.B * Kth;

    // update vt, Vxt, Vtt
    vc.vt = model.gamma + vn.vt;
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
  ALIGATOR_TRACY_ZONE_SCOPED;

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

    ConstRowMatrixRef A = d.fb.blockRow(2);
    ConstVectorRef a = d.ff.blockSegment(2);
    xs[t + 1].noalias() = a + A * xs[t];

    if (model.nth > 0 && theta_.has_value()) {
      ConstVectorRef theta = *theta_;
      ConstRowMatrixRef Ath = d.fth.blockRow(2);
      xs[t + 1].noalias() += Ath * theta;
    }

    const CostToGo &vn = datas[t + 1].vm;
    lbdas[t + 1].noalias() = vn.vx + vn.Vxx.transpose() * xs[t + 1];
    if (model.nth > 0 && theta_.has_value()) {
      lbdas[t + 1].noalias() += vn.Vxt * (*theta_);
    }
  }
  return true;
}

} // namespace gar
} // namespace aligator
