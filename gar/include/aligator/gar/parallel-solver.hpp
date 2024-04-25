#pragma once

#include "riccati-base.hpp"
#include "riccati-impl.hpp"

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
class ParallelRiccatiSolver : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using Base::datas;

  using Impl = ProximalRiccatiKernel<Scalar>;
  using KnotType = LQRKnotTpl<Scalar>;

  using BlkMat = BlkMatrix<MatrixXs, -1, -1>;
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;

  explicit ParallelRiccatiSolver(LQRProblemTpl<Scalar> &problem,
                                 const uint num_threads);

  void allocateLeg(uint start, uint end, bool last_leg);

  static void setupKnot(KnotType &knot, const Scalar mudyn) {
    ZoneScoped;
    ALIGATOR_NOMALLOC_BEGIN;
    knot.Gx = knot.A.transpose();
    knot.Gu = knot.B.transpose();
    knot.Gth.setZero();
    knot.Gth.diagonal().setConstant(-mudyn);
    knot.gamma = knot.f;
    ALIGATOR_NOMALLOC_END;
  }

  bool backward(const Scalar mudyn, const Scalar mueq);

  inline void collapseFeedback() {
    using RowMatrix = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
    StageFactor<Scalar> &d = datas[0];
    Eigen::Ref<RowMatrix> K = d.fb.blockRow(0);
    Eigen::Ref<RowMatrix> Kth = d.fth.blockRow(0);

    // condensedSystem.subdiagonal contains the 'U' factors in the
    // block-tridiag UDUt decomposition
    // and ∂Xi+1 = -Ui+1.t ∂Xi
    auto &Up1t = condensedKktSystem.subdiagonal[1];
    K.noalias() -= Kth * Up1t;
  }

  struct condensed_system_t {
    std::vector<MatrixXs> subdiagonal;
    std::vector<MatrixXs> diagonal;
    std::vector<MatrixXs> superdiagonal;
    std::vector<Eigen::BunchKaufman<MatrixXs>> facs;
  };

  /// @brief Create the sparse representation of the reduced KKT system.
  void assembleCondensedSystem(const Scalar mudyn);

  bool forward(VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
               VectorOfVectors &lbdas,
               const std::optional<ConstVectorRef> & = std::nullopt) const;

  /// Number of parallel divisions in the problem: \f$J+1\f$ in the math.
  uint numThreads;

  /// Hold the compressed representation of the condensed KKT system
  condensed_system_t condensedKktSystem;
  /// Contains the right-hand side and solution of the condensed KKT system.
  BlkVec condensedKktRhs;

  /// @brief Initialize the buffers for the block-tridiagonal system.
  void initializeTridiagSystem(const std::vector<long> &dims);

protected:
  LQRProblemTpl<Scalar> *problem_;
};

} // namespace gar
} // namespace aligator

#include "parallel-solver.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "parallel-solver.txx"
#endif
