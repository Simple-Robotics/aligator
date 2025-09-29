/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2O25 INRIA
#pragma once

#include "riccati-base.hpp"
#include "riccati-kernel.hpp"

namespace aligator {
namespace gar {

/// @brief A Riccati-like solver for the proximal LQ subproblem in ProxDDP.
template <typename _Scalar>
class ProximalRiccatiSolver : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using allocator_type = ::aligator::polymorphic_allocator;

  using Kernel = ProximalRiccatiKernel<Scalar>;
  using StageFactorType = typename Kernel::StageFactorType;
  using CostToGo = typename StageFactorType::CostToGo;
  using kkt0_t = typename Kernel::kkt0_t;
  using KnotType = LqrKnotTpl<Scalar>;

  explicit ProximalRiccatiSolver(const LqrProblemTpl<Scalar> &problem);

  /// Backward sweep.
  bool backward(const Scalar mueq);

  bool forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
               std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
               const std::optional<ConstVectorRef> &theta = std::nullopt) const;

  void cycleAppend(const KnotType &knot);
  VectorRef getFeedforward(size_t i) { return datas[i].ff.matrix(); }
  RowMatrixRef getFeedback(size_t i) { return datas[i].fb.matrix(); }

  allocator_type get_allocator() const { return problem_->get_allocator(); }

  std::pmr::vector<StageFactor<Scalar>> datas;
  kkt0_t kkt0;                  //< initial stage KKT system
  ArenaMatrix<VectorXs> thGrad; //< optimal value gradient wrt parameter
  ArenaMatrix<MatrixXs> thHess; //< optimal value Hessian wrt parameter

protected:
  const LqrProblemTpl<Scalar> *problem_;
};

template <typename Scalar>
ProximalRiccatiSolver(const LqrProblemTpl<Scalar> &)
    -> ProximalRiccatiSolver<Scalar>;

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class ProximalRiccatiSolver<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
