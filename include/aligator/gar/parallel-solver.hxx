/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/block-tridiagonal.hpp"
#include "aligator/gar/work.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/utils/mpc-util.hpp"
#include "aligator/tracy.hpp"
#include "aligator/utils/exceptions.hpp"

#include "aligator/threads.hpp"

namespace aligator::gar {

#ifdef ALIGATOR_MULTITHREADING
template <typename Scalar>
ParallelRiccatiSolver<Scalar>::ParallelRiccatiSolver(
    LqrProblemTpl<Scalar> &problem, const uint num_threads)
    : Base()
    , numThreads(num_threads)
    , problem_(&problem) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  if (num_threads < 2) {
    ALIGATOR_RUNTIME_ERROR(
        "({:s}) num_threads (={:d}) should be greater than or equal to 2.",
        __FUNCTION__, num_threads);
  }

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
  condensedKktSolution = condensedKktRhs;
  condensedErr = condensedKktRhs;
  initializeTridiagSystem(dims);

  assert(datas.size() == (N + 1));
}

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::allocateLeg(uint start, uint end,
                                                bool last_leg) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  for (uint t = start; t < end; t++) {
    KnotType &knot = problem_->stages[t];
    if (!last_leg)
      knot.addParameterization(knot.nx);
    datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nx2, knot.nth);
  }
}

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::assembleCondensedSystem(
    const Scalar mudyn) {
  ALIGATOR_TRACY_ZONE_SCOPED;
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
bool ParallelRiccatiSolver<Scalar>::backward(const Scalar mueq) {

  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_SCOPED_N("parallel_backward");
  auto N = static_cast<uint>(problem_->horizon());
  for (uint i = 0; i < numThreads - 1; i++) {
    uint end = get_work(N, i, numThreads).end;
    setupKnot(problem_->stages[end - 1]);
  }
  Eigen::setNbThreads(1);
  aligator::omp::set_default_options(numThreads, false);
#pragma omp parallel num_threads(numThreads)
  {
    uint i = (uint)omp::get_thread_id();
#ifdef __linux__
    char *thrdname = new char[16];
    snprintf(thrdname, 16, "thread%d[c%d]", int(i), sched_getcpu());
    ALIGATOR_TRACY_SET_THREAD_NAME(thrdname);
#endif
    auto [beg, end] = get_work(N, i, numThreads);
    boost::span<const KnotType> stview =
        make_span_from_indices(problem_->stages, beg, end);
    boost::span<StageFactor<Scalar>> dtview =
        make_span_from_indices(datas, beg, end);
    Kernel::backwardImpl(stview, mueq, dtview);
  }

  {
    Eigen::setNbThreads(0);
    assembleCondensedSystem(0.0);
    condensedKktSolution = condensedKktRhs;
    condensedFacs.diagonalFacs = condensedKktSystem.diagonal;
    condensedFacs.upFacs = condensedKktSystem.subdiagonal;

    // This routine may have accuracy issues
    symmetricBlockTridiagSolve(condensedKktSystem.subdiagonal,
                               condensedKktSystem.diagonal,
                               condensedKktSystem.superdiagonal,
                               condensedKktSolution, condensedFacs.ldlt);
    condensedFacs.diagonalFacs.swap(condensedKktSystem.diagonal);
    condensedFacs.upFacs.swap(condensedKktSystem.subdiagonal);

    // iterative refinement
    constexpr int maxRefinementSteps = 5;
    for (int i = 0; i < maxRefinementSteps; i++) {
      // 1. compute residual into rhs
      blockTridiagMatMul(condensedKktSystem.subdiagonal,
                         condensedKktSystem.diagonal,
                         condensedKktSystem.superdiagonal, condensedKktSolution,
                         condensedErr, -1.0);
      condensedErr.matrix() *= -1;

      Scalar resdl = math::infty_norm(condensedErr.matrix());
      if (resdl <= condensedThreshold)
        return true;

      // 2. perform refinement step and swap
      blockTridiagRefinementStep(condensedFacs.upFacs,
                                 condensedKktSystem.superdiagonal,
                                 condensedFacs.ldlt, condensedErr);
      condensedKktSolution.matrix() += condensedErr.matrix();
      condensedErr = condensedKktRhs.matrix();
    }
  }

  return true;
}

template <typename Scalar>
bool ParallelRiccatiSolver<Scalar>::forward(
    VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
    VectorOfVectors &lbdas, const std::optional<ConstVectorRef> &) const {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_SCOPED_N("parallel_forward");
  uint N = (uint)problem_->horizon();
  for (uint i = 0; i < numThreads; i++) {
    uint i0 = get_work(N, i, numThreads).beg;
    lbdas[i0] = condensedKktSolution[2 * i];
    xs[i0] = condensedKktSolution[2 * i + 1];
  }
  Eigen::setNbThreads(1);
  const auto &stages = problem_->stages;

#pragma omp parallel num_threads(numThreads)
  {
    uint i = (uint)omp::get_thread_id();
    auto [beg, end] = get_work(N, i, numThreads);
    boost::span xsview = make_span_from_indices(xs, beg, end);
    boost::span usview = make_span_from_indices(us, beg, end);
    boost::span vsview = make_span_from_indices(vs, beg, end);
    boost::span lsview = make_span_from_indices(lbdas, beg, end);
    boost::span stview = make_span_from_indices(stages, beg, end);
    boost::span dsview = make_span_from_indices(datas, beg, end);
    if (i < numThreads - 1) {
      Kernel::forwardImpl(stview, dsview, xsview, usview, vsview, lsview,
                          lbdas[end]);
    } else {
      Kernel::forwardImpl(stview, dsview, xsview, usview, vsview, lsview);
    }
  }
  Eigen::setNbThreads(0);
  return true;
}

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::cycleAppend(const KnotType &knot) {
  rotate_vec_left(datas, 0, 1);
  datas[problem_->horizon() - 1ul] =
      StageFactor<Scalar>(knot.nx, knot.nu, knot.nc, knot.nx2, knot.nth);
  rotate_vec_left(condensedKktSystem.subdiagonal);
  rotate_vec_left(condensedKktSystem.diagonal);
  rotate_vec_left(condensedKktSystem.superdiagonal);
  rotate_vec_left(condensedFacs.diagonalFacs);
  rotate_vec_left(condensedFacs.upFacs);
  rotate_vec_left(condensedFacs.ldlt);

  auto [i0, i1] = get_work(problem_->horizon(), numThreads - 2u, numThreads);
  uint dim0 = problem_->stages[i0].nx;
  uint dim1 = problem_->stages[i1 - 1u].nx;
  condensedKktSystem.subdiagonal.back().setZero(dim1, dim0);
  condensedKktSystem.diagonal.back().setZero(dim1, dim1);
  condensedFacs.diagonalFacs.back().setZero(dim1, dim1);
  condensedFacs.upFacs.back().setZero(dim1, dim1);
  condensedFacs.ldlt.back() = BunchKaufman<MatrixXs>(dim1);
};

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::initializeTridiagSystem(
    const std::vector<long> &dims) {
  ALIGATOR_TRACY_ZONE_SCOPED;

  condensedKktSystem.subdiagonal.reserve(dims.size() - 1);
  condensedKktSystem.diagonal.reserve(dims.size());
  condensedKktSystem.superdiagonal.reserve(dims.size() - 1);
  condensedFacs.diagonalFacs.reserve(dims.size());
  condensedFacs.upFacs.reserve(dims.size());
  condensedFacs.ldlt.reserve(dims.size());

  const auto emplace_factor = [](condensed_system_factor &f, Eigen::Index dim) {
    f.diagonalFacs.emplace_back(dim, dim);
    f.upFacs.emplace_back(dim, dim);
    f.ldlt.emplace_back(dim);
  };

  condensedKktSystem.diagonal.emplace_back(dims[0], dims[0]);
  emplace_factor(condensedFacs, dims[0]);

  for (uint i = 0; i < dims.size() - 1; i++) {
    condensedKktSystem.superdiagonal.emplace_back(dims[i], dims[i + 1]);
    condensedKktSystem.diagonal.emplace_back(dims[i + 1], dims[i + 1]);
    condensedKktSystem.subdiagonal.emplace_back(dims[i + 1], dims[i]);
    emplace_factor(condensedFacs, dims[i + 1]);
  }
}
#endif

} // namespace aligator::gar
