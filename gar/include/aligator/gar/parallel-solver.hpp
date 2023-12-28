#pragma once

#include "./riccati-impl.hpp"
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
/// TODO: implement tailored reduced system solver
/// TODO: generalize to more than 2 legs
template <typename _Scalar> class ParallelRiccatiSolver {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using Impl = ProximalRiccatiImpl<Scalar>;
  using StageFactor = typename Impl::StageFactor;
  using value_t = typename Impl::value_t;
  using KnotType = LQRKnotTpl<Scalar>;

  using BlkMat = BlkMatrix<MatrixXs, -1, -1>;
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;

  explicit ParallelRiccatiSolver(const LQRProblemTpl<Scalar> &problem)
      : datas(), problem(problem) {

    uint N = (uint)problem.horizon();
    uint i0 = 0;
    uint i1 = (N + 1) / 2;
    uint i2 = N + 1;
    splitIdx = {i0, i1, i2};

    build_leg(i0, i1);
    build_leg(i1, i2);

    assert(datas.size() == (N + 1));
    assert(checkIndices());
  }

  inline bool checkIndices() const {
    if (splitIdx[0] != 0)
      return false;

    for (size_t i = 0; i < splitIdx.size() - 1; i++) {
      if (splitIdx[i] >= splitIdx[i + 1])
        return false;
    }
    return true;
  }

  void build_leg(uint start, uint end) {
    bool last_leg = (int)end == problem.horizon() + 1;
    for (uint t = start; t < end; t++) {
      const KnotType &knot = problem.stages[t];
      if (!last_leg) {
        const_cast<KnotType &>(knot).addParameterization(knot.nx);
        assert(knot.nx == knot.nth);
      }
      datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nth);
    }
    if (!last_leg) {
      // last knot needs parameter to be set
      KnotType &knot = const_cast<KnotType &>(problem.stages[end - 1]);
      knot.Gx = knot.A.transpose();
      knot.Gu = knot.B.transpose();
      knot.gamma = knot.f;
    }
  }

  bool backward(Scalar mudyn, Scalar mueq) {

    bool ret = true;
#pragma omp parallel for reduction(& : ret)
    for (size_t i = 0; i < 2; i++) {
      boost::span<const KnotType> stview =
          make_span_from_indices(problem.stages, splitIdx[i], splitIdx[i + 1]);
      boost::span<StageFactor> dtview =
          make_span_from_indices(datas, splitIdx[i], splitIdx[i + 1]);
      bool r = Impl::backwardImpl(stview, mudyn, mueq, dtview);
      ret &= r;
    }
    solveReducedSystem(mudyn);
    return true;
  }

  /// Solve reduced KKT system using dense factorization.
  void solveReducedSystem(const Scalar mudyn) {
    auto i0 = splitIdx[0];
    auto i1 = splitIdx[1];
    const KnotType &kt0 = problem.stages[i0];
    const StageFactor &sf0 = datas[i0];
    const KnotType &kt1 = problem.stages[i1];
    const StageFactor &sf1 = datas[i1];

    std::vector<long> dims = {problem.nc0(), kt0.nx, kt1.nx, kt1.nx};
    // TODO: remove temporary memory allocation here
    BlkMat redKkt(dims, dims);
    redKkt.setZero();
    BlkVec redRhs(dims);

    redKkt(0, 0).diagonal().array() = -mudyn;
    redKkt(0, 1) = problem.G0;

    redKkt(1, 0) = problem.G0.transpose();
    redKkt(1, 1) = sf0.vm.Pmat;
    redKkt(1, 2) = sf0.vm.Vxt;

    redKkt(2, 1) = sf0.vm.Vxt.transpose();
    redKkt(2, 2) = sf0.vm.Vtt; // Pi0
    redKkt(2, 3) = kt0.E;

    redKkt(3, 2) = kt0.E.transpose();
    redKkt(3, 3) = sf1.vm.Pmat;

    redRhs[0] = problem.g0;
    redRhs[1] = sf0.vm.pvec;
    redRhs[2] = sf0.vm.vt;
    redRhs[3] = sf1.vm.pvec;

    redRhs.matrix() *= -1;

    Eigen::BunchKaufman<MatrixXs> chol{redKkt.matrix()};
    chol.solveInPlace(redRhs.matrix());
    reduced_kkt_sol.emplace(redRhs);
  }

  void forward(VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
               VectorOfVectors &lbdas) {
    const auto &rkkts = *reduced_kkt_sol;
    lbdas[0] = rkkts[0];
    xs[0] = rkkts[1];
    lbdas[splitIdx[1]] = rkkts[2];
    xs[splitIdx[1]] = rkkts[3];

    // #pragma omp parallel for
    for (size_t i = 0; i < 2; i++) {
      auto xsview = make_span_from_indices(xs, splitIdx[i], splitIdx[i + 1]);
      auto usview = make_span_from_indices(us, splitIdx[i], splitIdx[i + 1]);
      auto vsview = make_span_from_indices(vs, splitIdx[i], splitIdx[i + 1]);
      auto lsview = make_span_from_indices(lbdas, splitIdx[i], splitIdx[i + 1]);
      auto stview =
          make_span_from_indices(problem.stages, splitIdx[i], splitIdx[i + 1]);
      auto dsview = make_span_from_indices(datas, splitIdx[i], splitIdx[i + 1]);
      if (i == 1) {
        Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview);
      } else {
        ConstVectorRef theta1 = lbdas[splitIdx[i + 1]];
        Impl::forwardImpl(stview, dsview, xsview, usview, vsview, lsview,
                          theta1);
      }
    }
  }

  std::vector<StageFactor> datas;
  std::vector<uint> splitIdx;
  boost::optional<BlkVec> reduced_kkt_sol = boost::none;

protected:
  const LQRProblemTpl<Scalar> &problem;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class ParallelRiccatiSolver<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
