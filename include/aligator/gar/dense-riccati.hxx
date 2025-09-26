/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "dense-riccati.hpp"
#include "lqr-problem.hpp"
#include "aligator/utils/mpc-util.hpp"
#include "aligator/tracy.hpp"

namespace aligator::gar {

template <typename Scalar>
RiccatiSolverDense<Scalar>::RiccatiSolverDense(
    const LqrProblemTpl<Scalar> &problem)
    : Base()
    , problem_(&problem) {
  auto N = (uint)problem_->horizon();
  const auto &stages = problem_->stages;
  Pxx.resize(N + 1);
  Pxt.resize(N + 1);
  Ptt.resize(N + 1);
  px.resize(N + 1);
  pt.resize(N + 1);
  for (uint i = 0; i <= N; i++) {
    uint nx = stages[i].nx;
    uint nth = stages[i].nth;
    Pxx[i].setZero(nx, nx);
    Pxt[i].setZero(nx, nth);
    Ptt[i].setZero(nth, nth);
    px[i].setZero(nx);
    pt[i].setZero(nth);
    stage_factors.emplace_back(nx, stages[i].nu, stages[i].nc, stages[i].nx2,
                               nth);
  }

  uint nx0 = stages[0].nx;
  uint nth = stages[0].nth;
  std::array<long, 2> dims0 = {nx0, problem_->nc0()};
  kkt0 = {decltype(kkt0.mat)(dims0, dims0), decltype(kkt0.ff)(dims0, {1}),
          decltype(kkt0.fth)(dims0, {nth}),
          BunchKaufman<MatrixXs>(nx0 + problem_->nc0())};
  thGrad.setZero(nth);
  thHess.setZero(nth, nth);
}

template <typename Scalar>
bool RiccatiSolverDense<Scalar>::backward(const Scalar mudyn,
                                          const Scalar mueq) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  const auto &stages = problem_->stages;

  const uint N = (uint)problem_->horizon();
  Kernel::terminalSolve(stages[N], stage_factors[N],
                        {Pxx[N], Pxt[N], Ptt[N], px[N], pt[N]}, mueq);

  uint i = N - 1;
  while (true) {
    typename Kernel::value vn{Pxx[i + 1], Pxt[i + 1], Ptt[i + 1], px[i + 1],
                              pt[i + 1]};
    Kernel::stageKernelSolve(stages[i], stage_factors[i],
                             {Pxx[i], Pxt[i], Ptt[i], px[i], pt[i]}, &vn, mudyn,
                             mueq);

    if (i == 0)
      break;
    i--;
  }

  // initial stage
  Eigen::Map G0 = problem_->G0.to_const_map();
  Eigen::Map g0 = problem_->g0.to_const_map();
  kkt0.mat.setZero();
  kkt0.mat(0, 0) = Pxx[0];
  kkt0.mat(0, 1) = G0.transpose();
  kkt0.mat(1, 0) = G0;
  kkt0.mat(1, 1).diagonal().array() = -mudyn;
  kkt0.ldl.compute(kkt0.mat.matrix());

  kkt0.ff[0] = -px[0];
  kkt0.ff[1] = -g0;

  kkt0.fth.blockRow(0) = -Pxt[0];
  kkt0.fth.blockRow(1).setZero();

  kkt0.ldl.solveInPlace(kkt0.ff.matrix());
  kkt0.ldl.solveInPlace(kkt0.fth.matrix());

  thGrad.noalias() = pt[0] + Pxt[0].transpose() * kkt0.ff[0];
  thHess.noalias() = Ptt[0] + Pxt[0].transpose() * kkt0.fth.blockRow(0);

  return true;
}

template <typename Scalar>
bool RiccatiSolverDense<Scalar>::forward(
    std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
    std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
    const std::optional<ConstVectorRef> &theta_) const {
  ALIGATOR_NOMALLOC_SCOPED;
  xs[0] = kkt0.ff[0];
  lbdas[0] = kkt0.ff[1];
  if (theta_.has_value()) {
    xs[0].noalias() += kkt0.fth.blockRow(0) * theta_.value();
    lbdas[0].noalias() += kkt0.fth.blockRow(1) * theta_.value();
  }

  uint N = (uint)problem_->horizon();
  assert(xs.size() == N + 1);
  assert(us.size() >= N);
  assert(vs.size() == N + 1);
  assert(lbdas.size() == N + 1);
  for (uint i = 0; i <= N; i++) {
    const KnotType &model = problem_->stages[i];
    const Data &data = stage_factors[i];
    Kernel::forwardStep(i, i == N, model, data, xs, us, vs, lbdas, theta_);
  }
  return true;
}

template <typename Scalar>
void RiccatiSolverDense<Scalar>::cycleAppend(const KnotType &knot) {
  auto N = (uint)problem_->horizon();

  rotate_vec_left(stage_factors, 0, 1);

  rotate_vec_left(Pxx, 0, 1);
  rotate_vec_left(Pxt, 0, 1);
  rotate_vec_left(Ptt, 0, 1);
  rotate_vec_left(px, 0, 1);
  rotate_vec_left(pt, 0, 1);

  uint nx = knot.nx;
  uint nth = knot.nth;
  std::array<long, 4> dims = {knot.nu, knot.nc, knot.nx2, knot.nx2};
  long ntot = dims[0] + dims[1] + dims[2] + dims[3];

  Data &fN = stage_factors[N - 1];
  fN.setZero();
  // resize last ldl
  using ldl_t = std::decay_t<decltype(fN.ldl)>;
  fN.ldl = ldl_t(ntot);

  Pxx[N - 1].setZero(nx, nx);
  Pxt[N - 1].setZero(nx, nth);
  Ptt[N - 1].setZero(nth, nth);
  px[N - 1].setZero(nx);
  pt[N - 1].setZero(nth);
};

} // namespace aligator::gar
