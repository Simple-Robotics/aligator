#pragma once

#include "riccati-base.hpp"
#include "block-tridiagonal-solver.hpp"
#include "work.hpp"
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace aligator {
namespace gar {

/// A parallel-condensing LQ solver. This solver condenses the problem into a
/// reduced saddle-point problem in a subset of the states and costates,
/// corresponding to the time indices where the problem was split up.
/// These splitting variables are used to exploit the problem's
/// partially-separable structure: each "leg" is then condensed into its value
/// function with respect to both its initial state and last costate (linking to
/// the next leg). The saddle-point is cast into a linear system which is solved
/// by dense LDL factorization.
template <typename _Scalar>
class ParallelRiccatiSolver2 : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using Base::datas;

  using Impl = ProximalRiccatiKernel<Scalar>;
  using KnotType = LQRKnotTpl<Scalar>;

  using BlkMat = BlkMatrix<MatrixXs, -1, -1>;
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;

  ParallelRiccatiSolver2(LQRProblemTpl<Scalar> &problem, const uint num_threads)
      : Base(), numThreads(num_threads),
        global_(tbb::global_control::max_allowed_parallelism,
                (size_t)numThreads),
        problem_(&problem) {
    ZoneScoped;

    uint N = (uint)problem.horizon();
    for (uint i = 0; i < num_threads; i++) {
      auto [i0, i1] = get_work(N, i, num_threads);
      allocateLeg(i0, i1, i == (num_threads - 1));
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

  void allocateLeg(uint start, uint end, bool last_leg) {
    ZoneScoped;
    for (uint t = start; t < end; t++) {
      KnotType &knot = problem_->stages[t];
      if (!last_leg)
        knot.addParameterization(knot.nx);
      datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nth);
    }
    if (!last_leg) {
      // last knot in the leg needs parameter to be set
      setupKnot(problem_->stages[end - 1]);
    }
  }

  static void setupKnot(KnotType &knot) {
    ZoneScoped;
    ALIGATOR_NOMALLOC_BEGIN;
    knot.Gx = knot.A.transpose();
    knot.Gu = knot.B.transpose();
    knot.gamma = knot.f;
    ALIGATOR_NOMALLOC_END;
  }

  bool backward(const Scalar mudyn, const Scalar mueq) {

    ALIGATOR_NOMALLOC_BEGIN;
    ZoneScopedN("tbb_parallel_backward");
    Eigen::setNbThreads(1);
    auto &stages = problem_->stages;
    auto &datas = this->datas;

    uint N = static_cast<uint>(problem_->horizon());
    for (uint i = 0; i < numThreads - 1; i++) {
      auto [_, end] = get_work(N, i, numThreads);
      setupKnot(stages[end - 1]);
    }

    tbb::parallel_for(
        0U, numThreads,
        [N, numThreads = numThreads, mueq, mudyn, &stages, &datas](uint i) {
          char *thrdname = new char[16];
          int cpu = sched_getcpu();
          snprintf(thrdname, 16, "thread%d[c%d]", int(i), cpu);
          tracy::SetThreadName(thrdname);
          auto [beg, end] = get_work(N, i, numThreads);
          boost::span<const KnotType> stview =
              make_span_from_indices(stages, beg, end);
          boost::span<StageFactor<Scalar>> dtview =
              make_span_from_indices(datas, beg, end);
          Impl::backwardImpl(stview, mudyn, mueq, dtview);
          // end -= 1;
          // Impl::stageKernelSolve(stages[end], datas[end], nullptr, mudyn,
          // mueq); int _t; for (_t = int(end) - 1; _t >= int(beg); --_t) {
          //   auto t = static_cast<size_t>(_t);
          //   auto &vn = datas[t + 1].vm;
          //   Impl::stageKernelSolve(stages[t], datas[t], &vn, mudyn, mueq);
          // }
        });

    assembleCondensedSystem(mudyn);
    Eigen::setNbThreads(0);
    symmetricBlockTridiagSolve(condensedKktSystem.subdiagonal,
                               condensedKktSystem.diagonal,
                               condensedKktSystem.superdiagonal,
                               condensedKktRhs, condensedKktSystem.facs);

    ALIGATOR_NOMALLOC_END;
    return true;
  }

  struct condensed_system_t {
    std::vector<MatrixXs> subdiagonal;
    std::vector<MatrixXs> diagonal;
    std::vector<MatrixXs> superdiagonal;
    std::vector<Eigen::BunchKaufman<MatrixXs>> facs;
  };

  /// Create the sparse representation of the reduced KKT system.
  void assembleCondensedSystem(const Scalar mudyn) {
    ZoneScoped;
    std::vector<MatrixXs> &subdiagonal = condensedKktSystem.subdiagonal;
    std::vector<MatrixXs> &diagonal = condensedKktSystem.diagonal;
    std::vector<MatrixXs> &superdiagonal = condensedKktSystem.superdiagonal;

    const auto &stages = problem_->stages;
    uint N = static_cast<uint>(problem_->horizon());

    diagonal[0].setZero();
    diagonal[0].diagonal().setConstant(-mudyn);
    superdiagonal[0] = problem_->G0;

    diagonal[1] = datas[0].vm.Pmat;
    superdiagonal[1] = datas[0].vm.Vxt;

    // fill in for all legs
    for (uint i = 0; i < numThreads - 1; i++) {
      auto [i0, i1] = get_work(N, i, numThreads);

      size_t ip1 = i + 1;
      diagonal[2 * ip1] = datas[i0].vm.Vtt;
      diagonal[2 * ip1].diagonal().array() -= mudyn;

      diagonal[2 * ip1 + 1] = datas[i1].vm.Pmat;
      superdiagonal[2 * ip1] = stages[i1].E;

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
      size_t ip1 = i + 1;
      condensedKktRhs[2 * ip1] = -datas[i0].vm.vt;
      condensedKktRhs[2 * ip1 + 1] = -datas[i1].vm.pvec;
    }
  }

  bool forward(VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
               VectorOfVectors &lbdas,
               const std::optional<ConstVectorRef> & = std::nullopt) const {
    ALIGATOR_NOMALLOC_BEGIN;
    ZoneScopedN("tbb_parallel_forward");
    uint N = static_cast<uint>(problem_->horizon());
    for (uint i = 0; i < numThreads; i++) {
      auto [i0, _] = get_work(N, i, numThreads);
      lbdas[i0] = condensedKktRhs[2 * i];
      xs[i0] = condensedKktRhs[2 * i + 1];
    }

    auto &stages = problem_->stages;
    auto &datas = this->datas;
    tbb::parallel_for(0U, numThreads,
                      [N, numThreads = numThreads, &xs, &us, &vs, &lbdas,
                       &stages, &datas](uint i) {
                        // size_t i = omp::get_thread_id();
                        auto [beg, end] = get_work(N, i, numThreads);
                        auto xsview = make_span_from_indices(xs, beg, end);
                        auto usview = make_span_from_indices(us, beg, end);
                        auto vsview = make_span_from_indices(vs, beg, end);
                        auto lsview = make_span_from_indices(lbdas, beg, end);
                        auto stview = make_span_from_indices(stages, beg, end);
                        auto dsview = make_span_from_indices(datas, beg, end);
                        if (i < numThreads - 1) {
                          Impl::forwardImpl(stview, dsview, xsview, usview,
                                            vsview, lsview, lbdas[end]);
                        } else {
                          Impl::forwardImpl(stview, dsview, xsview, usview,
                                            vsview, lsview);
                        }
                      });
    ALIGATOR_NOMALLOC_END;
    return true;
  }

  /// Number of parallel divisions in the problem: \f$J+1\f$ in the math.
  uint numThreads;
  tbb::global_control global_;

  /// Hold the compressed representation of the condensed KKT system
  condensed_system_t condensedKktSystem;
  /// Contains the right-hand side and solution of the condensed KKT system.
  BlkVec condensedKktRhs;

  inline void initializeTridiagSystem(const std::vector<long> &dims) {
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

protected:
  LQRProblemTpl<Scalar> *problem_;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class ParallelRiccatiSolver2<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
