#pragma once

#include "./proximal-riccati.hpp"
#include "./lqr-problem.hpp"

#include "aligator/tracy.hpp"

namespace aligator::gar {

template <typename Scalar>
ProximalRiccatiSolver<Scalar>::ProximalRiccatiSolver(
    const LQRProblemTpl<Scalar> &problem)
    : Base(), kkt0(problem.stages[0].nx, problem.nc0(), problem.ntheta()),
      thGrad(problem.ntheta()), thHess(problem.ntheta(), problem.ntheta()),
      problem_(&problem) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  auto N = uint(problem_->horizon());
  auto &knots = problem_->stages;
  datas.reserve(N + 1);
  for (uint t = 0; t <= N; t++) {
    const KnotType &knot = knots[t];
    datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nx2, knot.nth);
  }
  thGrad.setZero();
  thHess.setZero();
  kkt0.mat.setZero();
}

template <typename Scalar>
bool ProximalRiccatiSolver<Scalar>::backward(const Scalar mudyn,
                                             const Scalar mueq) {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_NAMED(Zone1, true);
  bool ret = Impl::backwardImpl(problem_->stages, mudyn, mueq, datas);

  StageFactor<Scalar> &d0 = datas[0];
  value_t &vinit = d0.vm;
  vinit.Vxx = vinit.Pmat;
  vinit.vx = vinit.pvec;
  // initial stage
  {
    ALIGATOR_TRACY_ZONE_NAMED_N(Zone2, "factor_initial", true);
    kkt0.mat(0, 0) = vinit.Vxx;
    kkt0.mat(1, 0) = problem_->G0;
    kkt0.mat(0, 1) = problem_->G0.transpose();
    kkt0.mat(1, 1).diagonal().setConstant(-mudyn);
    kkt0.chol.compute(kkt0.mat.matrix());

    kkt0.ff.blockSegment(0) = -vinit.vx;
    kkt0.ff.blockSegment(1) = -problem_->g0;
    kkt0.chol.solveInPlace(kkt0.ff.matrix());
    kkt0.fth.blockRow(0) = -vinit.Vxt;
    kkt0.fth.blockRow(1).setZero();
    kkt0.chol.solveInPlace(kkt0.fth.matrix());

    thGrad.noalias() =
        vinit.vt + vinit.Vxt.transpose() * kkt0.ff.blockSegment(0);
    thHess.noalias() = vinit.Vtt + vinit.Vxt.transpose() * kkt0.fth.blockRow(0);
  }
  return ret;
}

template <typename Scalar>
bool ProximalRiccatiSolver<Scalar>::forward(
    std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
    std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
    const std::optional<ConstVectorRef> &theta_) const {
  ALIGATOR_TRACY_ZONE_SCOPED;

  // solve initial stage
  Impl::computeInitial(xs[0], lbdas[0], kkt0, theta_);

  return Impl::forwardImpl(problem_->stages, datas, xs, us, vs, lbdas, theta_);
}

} // namespace aligator::gar
