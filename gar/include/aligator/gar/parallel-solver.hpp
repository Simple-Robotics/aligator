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
  return boost::make_span(vec.data() + i0, i1 - i0);
}

/// @copybrief make_span_from_indices
template <class T>
boost::span<const T> make_span_from_indices(const std::vector<T> &vec,
                                            size_t i0, size_t i1) {
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
                                 const uint num_legs)
      : datas(), numLegs(num_legs), splitIdx(num_legs + 1), problem(problem) {

    uint N = (uint)problem.horizon();
    for (uint i = 0; i < num_legs; i++) {
      splitIdx[i] = i * (N + 1) / num_legs;
    }
    splitIdx[num_legs] = N + 1;
    for (uint i = 0; i < num_legs; i++) {
      buildLeg(splitIdx[i], splitIdx[i + 1], i == (num_legs - 1));
    }

    const std::vector<long> dims = compute_dims_for_reduced_system();
    condensedKktRhs = BlkVec(dims);
    condensedKktSystem = initialize_tridiag_system(dims);

    assert(datas.size() == (N + 1));
    assert(checkIndices());
  }

  auto compute_dims_for_reduced_system() const {
    const auto &stages = problem.stages;
    std::vector<long> dims{problem.nc0(), stages[0].nx}; // dims for rhs

    // fill in for all legs
    for (size_t i = 0; i < numLegs - 1; i++) {
      uint i0 = splitIdx[i];
      uint i1 = splitIdx[i + 1];
      dims.push_back(stages[i0].nx);
      dims.push_back(stages[i1 - 1].nx);
    }
    return dims;
  }

  inline bool checkIndices() const {
    if (splitIdx[0] != 0)
      return false;

    for (uint i = 0; i < numLegs; i++) {
      if (splitIdx[i] >= splitIdx[i + 1])
        return false;
    }
    return true;
  }

  void buildLeg(uint start, uint end, bool last_leg) {
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
    bool ret = true;
#pragma omp parallel for schedule(static, 1) num_threads(numLegs)              \
    reduction(& : ret)
    for (uint i = 0; i < numLegs; i++) {
      boost::span<const KnotType> stview =
          make_span_from_indices(problem.stages, splitIdx[i], splitIdx[i + 1]);
      boost::span<StageFactor<Scalar>> dtview =
          make_span_from_indices(datas, splitIdx[i], splitIdx[i + 1]);
      ret &= Impl::backwardImpl(stview, mudyn, mueq, dtview);
    }

    assembleCondensedSystem(mudyn);
    ALIGATOR_NOMALLOC_END;
    symmetric_block_tridiagonal_solve(
        condensedKktSystem.subdiagonal, condensedKktSystem.diagonal,
        condensedKktSystem.superdiagonal, condensedKktRhs);
    return true;
  }

  struct condensed_system_t {
    std::vector<MatrixXs> subdiagonal;
    std::vector<MatrixXs> diagonal;
    std::vector<MatrixXs> superdiagonal;
  };

  /// Create the sparse representation of the reduced KKT system.
  void assembleCondensedSystem(const Scalar mudyn) {
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
    for (size_t i = 0; i < numLegs - 1; i++) {
      uint i0 = splitIdx[i];
      uint i1 = splitIdx[i + 1];

      size_t ip1 = i + 1;
      diagonal[2 * ip1] = datas[i0].vm.Vtt;

      diagonal[2 * ip1 + 1] = datas[i1].vm.Pmat;
      superdiagonal[2 * ip1] = stages[i1].E;

      if (ip1 + 1 < numLegs) {
        superdiagonal[2 * ip1 + 1] = datas[i1].vm.Vxt;
      }
    }

    // fix sub diagonal
    for (size_t i = 0; i < subdiagonal.size(); i++) {
      subdiagonal[i] = superdiagonal[i].transpose();
    }

    condensedKktRhs[0] = -problem.g0;
    condensedKktRhs[1] = -datas[0].vm.pvec;

    for (size_t i = 0; i < numLegs - 1; i++) {
      uint i0 = splitIdx[i];
      uint i1 = splitIdx[i + 1];
      size_t ip1 = i + 1;
      condensedKktRhs[2 * ip1] = -datas[i0].vm.vt;
      condensedKktRhs[2 * ip1 + 1] = -datas[i1].vm.pvec;
    }
  }

  void forward(VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
               VectorOfVectors &lbdas) {
    for (size_t i = 0; i < numLegs; i++) {
      uint i0 = splitIdx[i];
      lbdas[i0] = condensedKktRhs[2 * i];
      xs[i0] = condensedKktRhs[2 * i + 1];
    }
    ALIGATOR_NOMALLOC_BEGIN;

#pragma omp parallel for schedule(static, 1) num_threads(numLegs)
    for (uint i = 0; i < numLegs; i++) {
      auto xsview = make_span_from_indices(xs, splitIdx[i], splitIdx[i + 1]);
      auto usview = make_span_from_indices(us, splitIdx[i], splitIdx[i + 1]);
      auto vsview = make_span_from_indices(vs, splitIdx[i], splitIdx[i + 1]);
      auto lsview = make_span_from_indices(lbdas, splitIdx[i], splitIdx[i + 1]);
      auto stview =
          make_span_from_indices(problem.stages, splitIdx[i], splitIdx[i + 1]);
      auto dsview = make_span_from_indices(datas, splitIdx[i], splitIdx[i + 1]);
      if (i == numLegs - 1) {
        Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview);
      } else {
        ConstVectorRef theta1 = lbdas[splitIdx[i + 1]];
        Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview,
                          theta1);
      }
    }
    ALIGATOR_NOMALLOC_END;
  }

  std::vector<StageFactor<Scalar>> datas;
  /// Number of parallel divisions in the problem: \f$J+1\f$ in the math.
  uint numLegs;
  /// Indices at which the problem should be split.
  std::vector<uint> splitIdx;

  /// Hold the compressed representation of the condensed KKT system
  condensed_system_t condensedKktSystem;
  /// Contains the right-hand side and solution of the condensed KKT system.
  BlkVec condensedKktRhs;

  LQRProblemTpl<Scalar> problem;

  inline static condensed_system_t
  initialize_tridiag_system(const std::vector<long> &dims) {
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
