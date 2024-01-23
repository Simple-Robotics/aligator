#pragma once

#include "./riccati-impl.hpp"
#include "./block-tridiagonal-solver.hpp"
#include "aligator/threads.hpp"

namespace aligator {
namespace gar {

/// Create a boost::span object from a vector and two indices.
template <class T>
boost::span<T> make_span_from_indices(std::vector<T> &vec, size_t i0,
                                      size_t i1) {
  ZoneScopedN("make_span");
  return boost::make_span(vec.data() + i0, i1 - i0);
}

/// @copybrief make_span_from_indices
template <class T>
boost::span<const T> make_span_from_indices(const std::vector<T> &vec,
                                            size_t i0, size_t i1) {
  ZoneScopedN("make_span_const");
  return boost::make_span(vec.data() + i0, i1 - i0);
}

/// A parallel-condensing LQ solver. This solver condenses the problem into a
/// reduced saddle-point problem in a subset of the states and costates,
/// corresponding to the time indices where the problem was split up.
/// These splitting variables are used to exploit the problem's
/// partially-separable structure: each "leg" is then condensed into its value
/// function with respect to both its initial state and last costate (linking to
/// the next leg). The saddle-point is cast into a linear system which is solved
/// by dense LDL factorization.
template <typename _Scalar> class ParallelRiccatiSolver {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using Impl = ProximalRiccatiImpl<Scalar>;
  using value_t = typename StageFactor<Scalar>::value_t;
  using KnotType = LQRKnotTpl<Scalar>;

  using BlkMat = BlkMatrix<MatrixXs, -1, -1>;
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;

  explicit ParallelRiccatiSolver(const LQRProblemTpl<Scalar> &problem,
                                 const uint num_threads)
      : datas(), numThreads(num_threads), splitIdx(num_threads + 1),
        problem(problem) {
    ZoneScoped;

    uint N = (uint)problem.horizon();
    for (uint i = 0; i < num_threads; i++) {
      splitIdx[i] = i * (N + 1) / num_threads;
    }
    splitIdx[num_threads] = N + 1;
    for (uint i = 0; i < num_threads; i++) {
      buildLeg(splitIdx[i], splitIdx[i + 1], i == (num_threads - 1));
    }

    std::vector<long> dims{problem.nc0(), problem.stages.front().nx};
    for (size_t i = 0; i < num_threads - 1; i++) {
      uint i0 = splitIdx[i];
      uint i1 = splitIdx[i + 1];
      dims.push_back(problem.stages[i0].nx);
      dims.push_back(problem.stages[i1 - 1].nx);
    }
    condensedKktRhs = BlkVec(dims);
    condensedKktSystem = initializeTridiagSystem(dims);

    assert(datas.size() == (N + 1));
    assert(checkIndices());
  }

  inline bool checkIndices() const {
    if (splitIdx[0] != 0)
      return false;

    for (uint i = 0; i < numThreads; i++) {
      if (splitIdx[i] >= splitIdx[i + 1])
        return false;
    }
    return true;
  }

  void buildLeg(uint start, uint end, bool last_leg) {
    ZoneScoped;
    for (uint t = start; t < end; t++) {
      KnotType &knot = problem.stages[t];
      if (!last_leg) {
        knot.addParameterization(knot.nx);
        assert(knot.nx == knot.nth);
      }
      datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nth);
    }
    if (!last_leg) {
      // last knot needs parameter to be set
      KnotType &knot = problem.stages[end - 1];
      knot.Gx = knot.A.transpose();
      knot.Gu = knot.B.transpose();
      knot.gamma = knot.f;
    }
  }

  bool backward(Scalar mudyn, Scalar mueq) {

    ALIGATOR_NOMALLOC_BEGIN;
    ZoneScopedN("parallel_backward");
    Eigen::setNbThreads(1);
    bool ret = true;
#pragma omp parallel num_threads(numThreads)
    {
      size_t id = omp::get_thread_id();
      char *thrdname = new char[16];
      int cpu = sched_getcpu();
      snprintf(thrdname, 16, "thread%d[c%d]", int(id), cpu);
      tracy::SetThreadName(thrdname);
#pragma omp for schedule(static, 1) reduction(& : ret)
      for (uint i = 0; i < numThreads; i++) {
        boost::span<const KnotType> stview = make_span_from_indices(
            problem.stages, splitIdx[id], splitIdx[i + 1]);
        boost::span<StageFactor<Scalar>> dtview =
            make_span_from_indices(datas, splitIdx[i], splitIdx[i + 1]);
        ret &= Impl::backwardImpl(stview, mudyn, mueq, dtview);
      }
    }

    Eigen::setNbThreads(0);
    assembleCondensedSystem(mudyn);
    ALIGATOR_NOMALLOC_END;
    ret &= symmetricBlockTridiagSolve(
        condensedKktSystem.subdiagonal, condensedKktSystem.diagonal,
        condensedKktSystem.superdiagonal, condensedKktRhs);
    return ret;
  }

  struct condensed_system_t {
    std::vector<MatrixXs> subdiagonal;
    std::vector<MatrixXs> diagonal;
    std::vector<MatrixXs> superdiagonal;
  };

  /// Create the sparse representation of the reduced KKT system.
  void assembleCondensedSystem(const Scalar mudyn) {
    ZoneScoped;
    std::vector<MatrixXs> &subdiagonal = condensedKktSystem.subdiagonal;
    std::vector<MatrixXs> &diagonal = condensedKktSystem.diagonal;
    std::vector<MatrixXs> &superdiagonal = condensedKktSystem.superdiagonal;

    const std::vector<KnotType> &stages = problem.stages;

    diagonal[0].setZero();
    diagonal[0].diagonal().setConstant(-mudyn);
    superdiagonal[0] = problem.G0;

    diagonal[1] = datas[0].vm.Pmat;
    superdiagonal[1] = datas[0].vm.Vxt;

    // fill in for all legs
    for (size_t i = 0; i < numThreads - 1; i++) {
      uint i0 = splitIdx[i];
      uint i1 = splitIdx[i + 1];

      size_t ip1 = i + 1;
      diagonal[2 * ip1] = datas[i0].vm.Vtt;

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

    condensedKktRhs[0] = -problem.g0;
    condensedKktRhs[1] = -datas[0].vm.pvec;

    for (size_t i = 0; i < numThreads - 1; i++) {
      uint i0 = splitIdx[i];
      uint i1 = splitIdx[i + 1];
      size_t ip1 = i + 1;
      condensedKktRhs[2 * ip1] = -datas[i0].vm.vt;
      condensedKktRhs[2 * ip1 + 1] = -datas[i1].vm.pvec;
    }
  }

  void forward(VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
               VectorOfVectors &lbdas) {
    ALIGATOR_NOMALLOC_BEGIN;
    ZoneScopedN("parallel_forward");
    for (size_t i = 0; i < numThreads; i++) {
      uint i0 = splitIdx[i];
      lbdas[i0] = condensedKktRhs[2 * i];
      xs[i0] = condensedKktRhs[2 * i + 1];
    }
    Eigen::setNbThreads(1);

#pragma omp parallel for schedule(static, 1) num_threads(numThreads)
    for (uint i = 0; i < numThreads; i++) {
      auto xsview = make_span_from_indices(xs, splitIdx[i], splitIdx[i + 1]);
      auto usview = make_span_from_indices(us, splitIdx[i], splitIdx[i + 1]);
      auto vsview = make_span_from_indices(vs, splitIdx[i], splitIdx[i + 1]);
      auto lsview = make_span_from_indices(lbdas, splitIdx[i], splitIdx[i + 1]);
      auto stview =
          make_span_from_indices(problem.stages, splitIdx[i], splitIdx[i + 1]);
      auto dsview = make_span_from_indices(datas, splitIdx[i], splitIdx[i + 1]);
      if (i == numThreads - 1) {
        Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview);
      } else {
        Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview,
                          lbdas[splitIdx[i + 1]]);
      }
    }
    Eigen::setNbThreads(0);
    ALIGATOR_NOMALLOC_END;
  }

  std::vector<StageFactor<Scalar>> datas;
  /// Number of parallel divisions in the problem: \f$J+1\f$ in the math.
  uint numThreads;
  /// Indices at which the problem should be split.
  std::vector<uint> splitIdx;

  /// Hold the compressed representation of the condensed KKT system
  condensed_system_t condensedKktSystem;
  /// Contains the right-hand side and solution of the condensed KKT system.
  BlkVec condensedKktRhs;

  /// An owned copy of the initial problem.
  LQRProblemTpl<Scalar> problem;

  inline static condensed_system_t
  initializeTridiagSystem(const std::vector<long> &dims) {
    ZoneScoped;
    std::vector<MatrixXs> subdiagonal;
    std::vector<MatrixXs> diagonal;
    std::vector<MatrixXs> superdiagonal;

    subdiagonal.reserve(dims.size() - 1);
    diagonal.reserve(dims.size());
    superdiagonal.reserve(dims.size());

    diagonal.emplace_back(dims[0], dims[0]);

    for (uint i = 0; i < dims.size() - 1; i++) {
      superdiagonal.emplace_back(dims[i], dims[i + 1]);
      diagonal.emplace_back(dims[i + 1], dims[i + 1]);
      subdiagonal.emplace_back(dims[i + 1], dims[i]);
    }

    return {std::move(subdiagonal), std::move(diagonal),
            std::move(superdiagonal)};
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class ParallelRiccatiSolver<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
