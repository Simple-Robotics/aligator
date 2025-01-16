/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./composite-costs.hpp"

namespace aligator {

/// @brief  Log-barrier of an underlying cost function.
template <typename Scalar>
struct RelaxedLogBarrierCostTpl : CostAbstractTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Base = CostAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  VectorXs barrier_weights_;
  xyz::polymorphic<StageFunction> residual_;
  Scalar threshold_;

  RelaxedLogBarrierCostTpl(xyz::polymorphic<Manifold> space,
                           xyz::polymorphic<StageFunction> function,
                           const ConstVectorRef &weight,
                           const Scalar threshold);

  RelaxedLogBarrierCostTpl(xyz::polymorphic<Manifold> space,
                           xyz::polymorphic<StageFunction> function,
                           const Scalar weight, const Scalar threshold);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostDataAbstract &data) const;

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostDataAbstract &data) const;

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostDataAbstract &data) const;

  shared_ptr<CostDataAbstract> createData() const {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData());
  }

  /// @brief Get a pointer to the underlying type of the residual, by attempting
  /// to cast.
  template <typename Derived> Derived *getResidual() {
    return dynamic_cast<Derived *>(&*residual_);
  }

  /// @copybrief getResidual().
  template <typename Derived> const Derived *getResidual() const {
    return dynamic_cast<const Derived *>(&*residual_);
  }
};

extern template struct RelaxedLogBarrierCostTpl<context::Scalar>;

} // namespace aligator

#include "./relaxed-log-barrier.hxx"
