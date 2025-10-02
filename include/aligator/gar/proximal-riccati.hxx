/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "proximal-riccati.hpp"
#include "lqr-problem.hpp"

#include "aligator/utils/mpc-util.hpp"
#include "aligator/tracy.hpp"

namespace aligator::gar {

template <typename Scalar>
ProximalRiccatiSolver<Scalar>::ProximalRiccatiSolver(
    const LqrProblemTpl<Scalar> &problem)
    : Base()
    , datas(problem.get_allocator())
    , kkt0(problem.stages[0].nx, problem.nc0(), problem.ntheta())
    , thGrad(problem.ntheta(), problem.get_allocator())
    , thHess(problem.ntheta(), problem.ntheta(), problem.get_allocator())
    , problem_(&problem) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  auto N = uint(problem_->horizon());
  datas.reserve(N + 1);
  for (uint t = 0; t <= N; t++) {
    const KnotType &knot = problem_->stages[t];
    datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nx2, knot.nth);
  }
  thGrad.setZero();
  thHess.setZero();
  kkt0.mat.setZero();
}

template <typename Scalar>
bool ProximalRiccatiSolver<Scalar>::backward(const Scalar mueq) {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_NAMED(Zone1, true);
  bool ret = Kernel::backwardImpl(problem_->stages, mueq, datas);

  StageFactor<Scalar> &d0 = datas[0];
  CostToGo &vinit = d0.vm;
  vinit.Vxx = vinit.Pmat;
  vinit.vx = vinit.pvec;
  // initial stage
  {
    ALIGATOR_TRACY_ZONE_NAMED_N(Zone2, "factor_initial", true);
    kkt0.mat(0, 0) = vinit.Vxx;
    kkt0.mat(1, 0) = problem_->G0;
    kkt0.mat(0, 1) = problem_->G0.transpose();
    kkt0.mat(1, 1).setZero();
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
  Kernel::computeInitial(xs[0], lbdas[0], kkt0, theta_);

  return Kernel::forwardImpl(problem_->stages, datas, xs, us, vs, lbdas,
                             theta_);
}

template <typename Scalar>
void ProximalRiccatiSolver<Scalar>::cycleAppend(const KnotType &knot) {
  rotate_vec_left(datas, 0, 1);
  uint N = uint(problem_->horizon() - 1);
  datas[N] = StageFactor<Scalar>(knot.nx, knot.nu, knot.nc, knot.nx2, knot.nth);
  thGrad.setZero();
  thHess.setZero();
  kkt0.mat.setZero();
};

} // namespace aligator::gar
