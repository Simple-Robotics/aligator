/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./composite-costs.hpp"

namespace aligator {

/// @brief  Log-barrier of an underlying cost function.
template <typename Scalar> struct LogResidualCostTpl : CostAbstractTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Base = CostAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  VectorXs barrier_weights_;
  shared_ptr<StageFunction> residual_;

  LogResidualCostTpl(shared_ptr<Manifold> space,
                     shared_ptr<StageFunction> function,
                     const ConstVectorRef &scale);

  LogResidualCostTpl(shared_ptr<Manifold> space,
                     shared_ptr<StageFunction> function, const Scalar scale);

  void configure(
      CommonModelBuilderContainer &common_buider_container) const override;

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

  shared_ptr<CostDataAbstract>
  createData(const CommonModelDataContainer &container) const override {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData(container));
  }
};

extern template struct LogResidualCostTpl<context::Scalar>;

} // namespace aligator

#include "./log-residual-cost.hxx"
