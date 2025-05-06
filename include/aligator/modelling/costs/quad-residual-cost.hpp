/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "./composite-costs.hpp"
#include <aligator/context.hpp>

namespace aligator {

/** @brief Quadratic composite of an underlying function.
 *
 * @details This is defined as
 * \f[
 *      c(x, u) \overset{\triangle}{=} \frac{1}{2} \|r(x, u)\|_W^2.
 * \f]
 */
template <typename _Scalar>
struct QuadraticResidualCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  MatrixXs weights_;
  xyz::polymorphic<StageFunction> residual_;
  bool gauss_newton = true;

  QuadraticResidualCostTpl(xyz::polymorphic<Manifold> space,
                           xyz::polymorphic<StageFunction> function,
                           const ConstMatrixRef &weights);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data_) const;

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data_) const;

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostData &data_) const;

  shared_ptr<CostData> createData() const {
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

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct QuadraticResidualCostTpl<context::Scalar>;
#endif

} // namespace aligator

#include "./quad-residual-cost.hxx"
