/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./composite-costs.hpp"

namespace aligator {
/**
 * @brief  Log-barrier of an underlying cost function.
 *
 * @details Defined as :
 * \f[
 *      c(x,u) = -w\ln(r(x,u))
 * \f]
 *
 * where :
 *    -\f(\ c(x,u) \f) is the resulting cost
 *    -\f(\ r(x,u) \f) is the input function
 *    -\f(\ w \f) is the input scale
 *
 */
template <typename Scalar> struct LogResidualCostTpl : CostAbstractTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Base = CostAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  VectorXs barrier_weights_;
  xyz::polymorphic<StageFunction> residual_;

  LogResidualCostTpl(xyz::polymorphic<Manifold> space,
                     xyz::polymorphic<StageFunction> function,
                     const ConstVectorRef &scale);

  LogResidualCostTpl(xyz::polymorphic<Manifold> space,
                     xyz::polymorphic<StageFunction> function,
                     const Scalar scale);

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

extern template struct LogResidualCostTpl<context::Scalar>;

} // namespace aligator

#include "./log-residual-cost.hxx"
