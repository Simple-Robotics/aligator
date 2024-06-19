/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./composite-costs.hpp"

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
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  MatrixXs weights_;
  shared_ptr<StageFunction> residual_;
  bool gauss_newton = true;

  QuadraticResidualCostTpl(shared_ptr<Manifold> space,
                           shared_ptr<StageFunction> function,
                           const ConstMatrixRef &weights);

  void configure(
      CommonModelBuilderContainer &common_buider_container) const override;

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

  shared_ptr<CostData>
  createData(const CommonModelDataContainer &container) const override {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData(container));
  }
};

extern template struct QuadraticResidualCostTpl<context::Scalar>;

} // namespace aligator

#include "./quad-residual-cost.hxx"
