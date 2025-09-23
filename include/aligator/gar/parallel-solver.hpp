/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/gar/riccati-base.hpp"
#include "aligator/gar/riccati-kernel.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/tracy.hpp"

namespace aligator {
namespace gar {

#ifdef ALIGATOR_MULTITHREADING
/// @brief A parallel-condensing LQ solver.
/// @details This solver condenses the problem into a
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
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using StageFactorVec = std::vector<StageFactor<Scalar>>;
  StageFactorVec datas;

  using Kernel = ProximalRiccatiKernel<Scalar>;
  using KnotType = LqrKnotTpl<Scalar>;

  using BlkMat = BlkMatrix<MatrixXs, -1, -1>;
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;

  explicit ParallelRiccatiSolver(LqrProblemTpl<Scalar> &problem,
                                 const uint num_threads);

  void allocateLeg(uint start, uint end, bool last_leg);

  static void setupKnot(KnotType &knot, const Scalar mudyn) {
    ALIGATOR_TRACY_ZONE_SCOPED;
    ALIGATOR_NOMALLOC_SCOPED;
    knot.Gx = knot.A.transpose();
    knot.Gu = knot.B.transpose();
    knot.Gth.setZero();
    knot.Gth.diagonal().setConstant(-mudyn);
    knot.gamma = knot.f;
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
  };

  struct condensed_system_factor {
    std::vector<MatrixXs> diagonalFacs; //< diagonal factors
    std::vector<MatrixXs> upFacs;       //< transposed U factors
    std::vector<BunchKaufman<MatrixXs>> ldlt;
  };

  /// @brief Create the sparse representation of the reduced KKT system.
  void assembleCondensedSystem(const Scalar mudyn);

  bool forward(VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
               VectorOfVectors &lbdas,
               const std::optional<ConstVectorRef> & = std::nullopt) const;

  void cycleAppend(const KnotType &knot);
  VectorRef getFeedforward(size_t i) { return datas[i].ff.matrix(); }
  RowMatrixRef getFeedback(size_t i) { return datas[i].fb.matrix(); }

  /// Number of parallel divisions in the problem: \f$J+1\f$ in the math.
  uint numThreads;

  /// Hold the compressed representation of the condensed KKT system
  condensed_system_t condensedKktSystem;
  /// Condensed KKT system factors.
  condensed_system_factor condensedFacs;
  /// Contains the right-hand side and solution of the condensed KKT system.
  BlkVec condensedKktRhs, condensedKktSolution, condensedErr;

  Scalar condensedThreshold{1e-11};

  /// @brief Initialize the buffers for the block-tridiagonal system.
  void initializeTridiagSystem(const std::vector<long> &dims);

protected:
  LqrProblemTpl<Scalar> *problem_;
};

template <typename Scalar>
ParallelRiccatiSolver(LqrProblemTpl<Scalar> &, const uint)
    -> ParallelRiccatiSolver<Scalar>;
#endif

} // namespace gar
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/gar/parallel-solver.txx"
#endif
