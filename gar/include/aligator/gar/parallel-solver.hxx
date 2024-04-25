#pragma once

#include "parallel-solver.hpp"
#include "block-tridiagonal.hpp"
#include "work.hpp"

#include "aligator/threads.hpp"

namespace aligator::gar {

template <typename Scalar>
ParallelRiccatiSolver<Scalar>::ParallelRiccatiSolver(
    LQRProblemTpl<Scalar> &problem, const uint num_threads)
    : Base(), numThreads(num_threads), problem_(&problem) {
  ZoneScoped;

  uint N = (uint)problem.horizon();
  for (uint i = 0; i < num_threads; i++) {
    auto [start, end] = get_work(N, i, num_threads);
    allocateLeg(start, end, i == (num_threads - 1));
  }

  std::vector<long> dims{problem.nc0(), problem.stages.front().nx};
  for (uint i = 0; i < num_threads - 1; i++) {
    auto [i0, i1] = get_work(N, i, num_threads);
    dims.push_back(problem.stages[i0].nx);
    dims.push_back(problem.stages[i1 - 1].nx);
  }
  condensedKktRhs = BlkVec(dims);
  initializeTridiagSystem(dims);

  assert(datas.size() == (N + 1));
}

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::allocateLeg(uint start, uint end,
                                                bool last_leg) {
  ZoneScoped;
  for (uint t = start; t < end; t++) {
    KnotType &knot = problem_->stages[t];
    if (!last_leg)
      knot.addParameterization(knot.nx);
    datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nth);
  }
}

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::assembleCondensedSystem(
    const Scalar mudyn) {
  ZoneScoped;
  std::vector<MatrixXs> &subdiagonal = condensedKktSystem.subdiagonal;
  std::vector<MatrixXs> &diagonal = condensedKktSystem.diagonal;
  std::vector<MatrixXs> &superdiagonal = condensedKktSystem.superdiagonal;

  const auto &stages = problem_->stages;

  diagonal[0].setZero();
  diagonal[0].diagonal().setConstant(-mudyn);
  superdiagonal[0] = problem_->G0;

  diagonal[1] = datas[0].vm.Pmat;
  superdiagonal[1] = datas[0].vm.Vxt;

  uint N = (uint)problem_->horizon();
  // fill in for all legs
  for (uint i = 0; i < numThreads - 1; i++) {
    auto [i0, i1] = get_work(N, i, numThreads);
    uint ip1 = i + 1;
    diagonal[2 * ip1] = datas[i0].vm.Vtt;

    diagonal[2 * ip1 + 1] = datas[i1].vm.Pmat;
    superdiagonal[2 * ip1] = stages[i1 - 1].E;

    if (ip1 + 1 < numThreads) {
      superdiagonal[2 * ip1 + 1] = datas[i1].vm.Vxt;
    }
  }

  // fix sub diagonal
  for (size_t i = 0; i < subdiagonal.size(); i++) {
    subdiagonal[i] = superdiagonal[i].transpose();
  }

  condensedKktRhs[0] = -problem_->g0;
  condensedKktRhs[1] = -datas[0].vm.pvec;

  for (uint i = 0; i < numThreads - 1; i++) {
    auto [i0, i1] = get_work(N, i, numThreads);
    uint ip1 = i + 1;
    condensedKktRhs[2 * ip1] = -datas[i0].vm.vt;
    condensedKktRhs[2 * ip1 + 1] = -datas[i1].vm.pvec;
  }
}

template <typename Scalar>
bool ParallelRiccatiSolver<Scalar>::backward(const Scalar mudyn,
                                             const Scalar mueq) {

  ALIGATOR_NOMALLOC_BEGIN;
  ZoneScopedN("parallel_backward");
  auto N = static_cast<uint>(problem_->horizon());
  for (uint i = 0; i < numThreads - 1; i++) {
    uint end = get_work(N, i, numThreads).end;
    setupKnot(problem_->stages[end - 1], mudyn);
  }
  Eigen::setNbThreads(1);
  aligator::omp::set_default_options(numThreads, false);
#pragma omp parallel num_threads(numThreads)
  {
    uint i = (uint)omp::get_thread_id();
    char *thrdname = new char[16];
    int cpu = sched_getcpu();
    snprintf(thrdname, 16, "thread%d[c%d]", int(i), cpu);
    tracy::SetThreadName(thrdname);
    auto [beg, end] = get_work(N, i, numThreads);
    boost::span<const KnotType> stview =
        make_span_from_indices(problem_->stages, beg, end);
    boost::span<StageFactor<Scalar>> dtview =
        make_span_from_indices(datas, beg, end);
    Impl::backwardImpl(stview, mudyn, mueq, dtview);
#pragma omp barrier
#pragma omp single
    {
      Eigen::setNbThreads(0);
      assembleCondensedSystem(mudyn);
      symmetricBlockTridiagSolve(condensedKktSystem.subdiagonal,
                                 condensedKktSystem.diagonal,
                                 condensedKktSystem.superdiagonal,
                                 condensedKktRhs, condensedKktSystem.facs);
    }
  }

  ALIGATOR_NOMALLOC_END;
  return true;
}

template <typename Scalar>
bool ParallelRiccatiSolver<Scalar>::forward(
    VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
    VectorOfVectors &lbdas, const std::optional<ConstVectorRef> &) const {
  ALIGATOR_NOMALLOC_BEGIN;
  ZoneScopedN("parallel_forward");
  uint N = (uint)problem_->horizon();
  for (uint i = 0; i < numThreads; i++) {
    uint i0 = get_work(N, i, numThreads).beg;
    lbdas[i0] = condensedKktRhs[2 * i];
    xs[i0] = condensedKktRhs[2 * i + 1];
  }
  Eigen::setNbThreads(1);
  const auto &stages = problem_->stages;

#pragma omp parallel num_threads(numThreads)
  {
    uint i = (uint)omp::get_thread_id();
    auto [beg, end] = get_work(N, i, numThreads);
    auto xsview = make_span_from_indices(xs, beg, end);
    auto usview = make_span_from_indices(us, beg, end);
    auto vsview = make_span_from_indices(vs, beg, end);
    auto lsview = make_span_from_indices(lbdas, beg, end);
    auto stview = make_span_from_indices(stages, beg, end);
    auto dsview = make_span_from_indices(datas, beg, end);
    if (i < numThreads - 1) {
      Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview,
                        lbdas[end]);
    } else {
      Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview);
    }
  }
  Eigen::setNbThreads(0);
  ALIGATOR_NOMALLOC_END;
  return true;
}

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::initializeTridiagSystem(
    const std::vector<long> &dims) {
  ZoneScoped;
  std::vector<MatrixXs> subdiagonal;
  std::vector<MatrixXs> diagonal;
  std::vector<MatrixXs> superdiagonal;

  condensedKktSystem.subdiagonal.reserve(dims.size() - 1);
  condensedKktSystem.diagonal.reserve(dims.size());
  condensedKktSystem.superdiagonal.reserve(dims.size() - 1);
  condensedKktSystem.facs.reserve(dims.size());

  condensedKktSystem.diagonal.emplace_back(dims[0], dims[0]);
  condensedKktSystem.facs.emplace_back(dims[0]);

  for (uint i = 0; i < dims.size() - 1; i++) {
    condensedKktSystem.superdiagonal.emplace_back(dims[i], dims[i + 1]);
    condensedKktSystem.diagonal.emplace_back(dims[i + 1], dims[i + 1]);
    condensedKktSystem.subdiagonal.emplace_back(dims[i + 1], dims[i]);
    condensedKktSystem.facs.emplace_back(dims[i + 1]);
  }
}

} // namespace aligator::gar
