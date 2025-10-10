/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/gar/riccati-base.hpp"
#include "aligator/gar/riccati-kernel.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/tracy.hpp"

#ifdef ALIGATOR_MULTITHREADING
namespace aligator {
namespace gar {

/// @brief A parallel-condensing LQ solver.
/// @details This solver condenses the problem into a
/// reduced saddle-point problem in a subset of the states and costates,
/// corresponding to the time indices where the problem was split up.
/// These splitting variables are used to exploit the problem's
/// partially-separable structure: each "leg" is then condensed into its value
/// function with respect to both its initial state and last costate (linking to
/// the next leg). The saddle-point is cast into a linear system which is solved
/// by dense LDL factorization.
/// This allows parallel resolution of a (long) linear subproblem on multiple
/// CPU cores.
template <typename _Scalar>
class ParallelRiccatiSolver : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using Kernel = ProximalRiccatiKernel<Scalar>;
  using KnotType = LqrKnotTpl<Scalar>;
  using BlkView = BlkMatrix<VectorRef, -1, 1>;
  using allocator_type = ::aligator::polymorphic_allocator;

  explicit ParallelRiccatiSolver(LqrProblemTpl<Scalar> &problem,
                                 const uint num_threads);

  bool backward(const Scalar mueq) override;

  inline void collapseFeedback() override {
    StageFactor<Scalar> &d = datas[0];
    RowMatrixRef K = d.fb.blockRow(0);
    RowMatrixRef Kth = d.fth.blockRow(0);

    // condensedSystem.subdiagonal contains the 'U' factors in the
    // block-tridiag UDUt decomposition
    // and ∂Xi+1 = -Ui+1.t ∂Xi
    auto &Up1t = condensedKktSystem.subdiagonal[1];
    K.noalias() -= Kth * Up1t;
  }

  struct CondensedKkt {
    using ArMat = ArenaMatrix<MatrixXs>;
    std::pmr::vector<ArMat> subdiagonal;
    std::pmr::vector<ArMat> diagonal;
    std::pmr::vector<ArMat> superdiagonal;
    // factors
    std::pmr::vector<ArMat> diagonalFacs; //< diagonal factors
    std::pmr::vector<ArMat> upFacs;       //< transposed U factors
    std::pmr::vector<BunchKaufman<ArMat>> ldlt;
    explicit CondensedKkt(const allocator_type &alloc)
        : subdiagonal{alloc}
        , diagonal{alloc}
        , superdiagonal{alloc}
        , diagonalFacs{alloc}
        , upFacs{alloc}
        , ldlt{alloc} {}
  };

  /// @brief Create the sparse representation of the reduced KKT system.
  void assembleCondensedSystem(const Scalar mudyn);

  bool
  forward(VectorOfVectors &xs, VectorOfVectors &us, VectorOfVectors &vs,
          VectorOfVectors &lbdas,
          const std::optional<ConstVectorRef> & = std::nullopt) const override;

  void cycleAppend(const KnotType &knot) override;
  VectorRef getFeedforward(size_t i) override { return datas[i].ff.matrix(); }
  RowMatrixRef getFeedback(size_t i) override { return datas[i].fb.matrix(); }

  allocator_type get_allocator() const { return problem_->get_allocator(); }

  std::pmr::vector<StageFactor<Scalar>> datas;

  /// Block-sparse condensed KKT system
  CondensedKkt condensedKktSystem;
  /// Contains the right-hand side and solution of the condensed KKT system.
  ArenaMatrix<VectorXs> condensedKktRhs, condensedKktSolution, condensedErr;
  /// Tolerance on condensed KKT system
  Scalar condensedThreshold = 1e-10;
  /// Max number of refinement steps (condensed solver)
  uint maxRefinementSteps = 5u;

  /// Number of parallel divisions in the problem: \f$J+1\f$ in the math.
  uint getNumThreads() const noexcept { return numThreads; }

  /// @brief Initialize the buffers for the block-tridiagonal system.
  void initializeTridiagSystem();

protected:
  uint numThreads;
  LqrProblemTpl<Scalar> *problem_;
  std::vector<long> rhsDims_;

  void initialize();
};

template <typename Scalar>
ParallelRiccatiSolver(LqrProblemTpl<Scalar> &, const uint)
    -> ParallelRiccatiSolver<Scalar>;

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class ParallelRiccatiSolver<context::Scalar>;
#endif
} // namespace gar
} // namespace aligator
#endif
