/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/block-tridiagonal.hpp"
#include "aligator/gar/lqr-problem.hpp"
// #include "aligator/utils/mpc-util.hpp"
#include "aligator/tracy.hpp"
#include "aligator/utils/exceptions.hpp"
#include "aligator/threads.hpp"

#include <numeric>

namespace aligator::gar {
struct workrange_t {
  uint beg;
  uint end;
};

/// @brief Get a balanced work range corresponding to a horizon @p horz, thread
/// ID @p tid, and number of threads @p num_threads.
constexpr workrange_t get_work(uint horz, uint thread_id, uint num_threads) {
  uint start = thread_id * (horz + 1) / num_threads;
  uint stop = (thread_id + 1) * (horz + 1) / num_threads;
  assert(stop <= horz + 1);
  return {start, stop};
}

#ifdef ALIGATOR_MULTITHREADING
template <typename Scalar>
ParallelRiccatiSolver<Scalar>::ParallelRiccatiSolver(
    LqrProblemTpl<Scalar> &problem, const uint num_threads)
    : Base()
    , condensedKktSystem(problem.get_allocator())
    , condensedKktRhs(problem.get_allocator())
    , condensedKktSolution(problem.get_allocator())
    , condensedErr(problem.get_allocator())
    , numThreads(num_threads)
    , problem_(&problem) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  if (numThreads < 2) {
    ALIGATOR_RUNTIME_ERROR(
        "({:s}) numThreads ({:d}) should be greater than or equal to 2.",
        __FUNCTION__, numThreads);
  }

  this->initialize();
}

template <typename Scalar> void ParallelRiccatiSolver<Scalar>::initialize() {
  const auto allocate_leg = [this](uint start, uint end, bool last_leg) {
    uint nth = problem_->stages[end - 1].nx2;
    for (uint t = start; t < end; t++) {
      KnotType &knot = problem_->stages[t];
      if (!last_leg)
        knot.addParameterization(nth);
      datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nx2, knot.nth);
    }
  };

  const uint N = (uint)problem_->horizon();
  for (uint i = 0; i < numThreads; i++) {
    auto [i0, i1] = get_work(N, i, numThreads);
    allocate_leg(i0, i1, i == (numThreads - 1));
  }

  rhsDims_ = {problem_->nc0(), problem_->stages[0].nx};
  for (uint i = 0; i < numThreads - 1; i++) {
    auto [i0, i1] = get_work(N, i, numThreads);
    rhsDims_.push_back(problem_->stages[i0].nx);
    rhsDims_.push_back(problem_->stages[i1 - 1].nx);
  }
  long condensed_total_dim =
      std::accumulate(rhsDims_.begin(), rhsDims_.end(), 0l);
  condensedKktRhs.setZero(condensed_total_dim);
  condensedKktSolution = condensedKktRhs;
  condensedErr = condensedKktRhs;
  initializeTridiagSystem();

  assert(datas.size() == (N + 1));
}

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::assembleCondensedSystem(
    const Scalar mudyn) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  boost::span subdiagonal = condensedKktSystem.subdiagonal;
  boost::span diagonal = condensedKktSystem.diagonal;
  boost::span superdiagonal = condensedKktSystem.superdiagonal;

  diagonal[0].setZero();
  diagonal[0].diagonal().setConstant(-mudyn);
  superdiagonal[0] = problem_->G0;

  diagonal[1] = datas[0].vm.Vxx;
  superdiagonal[1] = datas[0].vm.Vxt;

  uint N = (uint)problem_->horizon();
  // fill in for all legs
  for (uint i = 0; i < numThreads - 1; i++) {
    auto [i0, i1] = get_work(N, i, numThreads);
    uint ip1 = i + 1;
    diagonal[2 * ip1] = datas[i0].vm.Vtt;

    diagonal[2 * ip1 + 1] = datas[i1].vm.Vxx;
    superdiagonal[2 * ip1].setIdentity() *= -1;

    if (ip1 + 1 < numThreads) {
      superdiagonal[2 * ip1 + 1] = datas[i1].vm.Vxt;
    }
  }

  // fix sub diagonal
  for (size_t i = 0; i < subdiagonal.size(); i++) {
    subdiagonal[i] = superdiagonal[i].transpose();
  }

  BlkView rhs_view{condensedKktRhs, rhsDims_};
  rhs_view.blockSegment(0) = -problem_->g0;
  rhs_view.blockSegment(1) = -datas[0].vm.vx;

  for (uint i = 0; i < numThreads - 1; i++) {
    auto [i0, i1] = get_work(N, i, numThreads);
    uint ip1 = i + 1;
    rhs_view.blockSegment(2 * ip1) = -datas[i0].vm.vt;
    rhs_view.blockSegment(2 * ip1 + 1) = -datas[i1].vm.vx;
  }
}

template <typename Scalar>
bool ParallelRiccatiSolver<Scalar>::backward(const Scalar mueq) {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_SCOPED_N("parallel_backward");

  const auto configure_knot = [](KnotType &knot) {
    knot.Gx = knot.A.transpose();
    knot.Gu = knot.B.transpose();
    knot.Gth.setZero();
    knot.gamma = knot.f;
  };

  const uint N = static_cast<uint>(problem_->horizon());
  for (uint i = 0; i < numThreads - 1; i++) {
    uint end = get_work(N, i, numThreads).end;
    configure_knot(problem_->stages[end - 1]);
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

  using ArMat = ArenaMatrix<MatrixXs>;
  {
    Eigen::setNbThreads(0);
    assembleCondensedSystem(0.0);
    condensedKktSolution = condensedKktRhs;
    condensedKktSystem.diagonalFacs = condensedKktSystem.diagonal;
    condensedKktSystem.upFacs = condensedKktSystem.subdiagonal;

    // This routine may have accuracy issues
    BlkView kkt_sol_view{condensedKktSolution, rhsDims_};
    symmetricBlockTridiagSolve<ArMat>(condensedKktSystem.subdiagonal,
                                      condensedKktSystem.diagonal,
                                      condensedKktSystem.superdiagonal,
                                      kkt_sol_view, condensedKktSystem.ldlt);
    condensedKktSystem.diagonalFacs.swap(condensedKktSystem.diagonal);
    condensedKktSystem.upFacs.swap(condensedKktSystem.subdiagonal);

    // iterative refinement
    BlkView err_rhs_view{condensedErr, rhsDims_}; // blocked view of the rhs
    for (uint i = 0; i < maxRefinementSteps; i++) {
      // 1. compute residual into rhs
      blockTridiagMatMul<ArMat>(
          condensedKktSystem.subdiagonal, condensedKktSystem.diagonal,
          condensedKktSystem.superdiagonal, kkt_sol_view, err_rhs_view, -1.0);
      condensedErr *= -1;

      Scalar resdl = math::infty_norm(condensedErr);
      if (resdl <= condensedThreshold)
        return true;

      // 2. perform refinement step and swap
      blockTridiagRefinementStep<ArMat>(condensedKktSystem.upFacs,
                                        condensedKktSystem.superdiagonal,
                                        condensedKktSystem.ldlt, err_rhs_view);
      condensedKktSolution += condensedErr;
      condensedErr = condensedKktRhs;
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
  BlkView sol_view{condensedKktSolution, rhsDims_};
  for (uint i = 0; i < numThreads; i++) {
    uint i0 = get_work(N, i, numThreads).beg;
    lbdas[i0] = sol_view.blockSegment(2 * i);
    xs[i0] = sol_view.blockSegment(2 * i + 1);
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
void ParallelRiccatiSolver<Scalar>::cycleAppend(const KnotType &) {
  datas.clear();
  condensedKktSystem.subdiagonal.clear();
  condensedKktSystem.diagonal.clear();
  condensedKktSystem.superdiagonal.clear();
  condensedKktSystem.diagonalFacs.clear();
  condensedKktSystem.upFacs.clear();
  condensedKktSystem.ldlt.clear();
  problem_->addParameterization(0);
  // just... reinitialise everything
  this->initialize();
};

template <typename Scalar>
void ParallelRiccatiSolver<Scalar>::initializeTridiagSystem() {
  ALIGATOR_TRACY_ZONE_SCOPED;
  const auto &dims = rhsDims_;

  condensedKktSystem.subdiagonal.reserve(dims.size() - 1);
  condensedKktSystem.diagonal.reserve(dims.size());
  condensedKktSystem.superdiagonal.reserve(dims.size() - 1);
  condensedKktSystem.diagonalFacs.reserve(dims.size());
  condensedKktSystem.upFacs.reserve(dims.size());
  condensedKktSystem.ldlt.reserve(dims.size());

  const auto emplace_factor = [](CondensedKkt &f, Eigen::Index dim) {
    f.diagonalFacs.emplace_back(dim, dim);
    f.upFacs.emplace_back(dim, dim);
    f.ldlt.emplace_back(dim);
  };

  condensedKktSystem.diagonal.emplace_back(dims[0], dims[0]);
  emplace_factor(condensedKktSystem, dims[0]);

  for (uint i = 0; i < dims.size() - 1; i++) {
    condensedKktSystem.superdiagonal.emplace_back(dims[i], dims[i + 1]);
    condensedKktSystem.diagonal.emplace_back(dims[i + 1], dims[i + 1]);
    condensedKktSystem.subdiagonal.emplace_back(dims[i + 1], dims[i]);
    emplace_factor(condensedKktSystem, dims[i + 1]);
  }
}
#endif

} // namespace aligator::gar
