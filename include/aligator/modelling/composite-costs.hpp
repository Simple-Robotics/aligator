/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/function-abstract.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/modelling/state-error.hpp"

#include <fmt/ostream.h>

namespace aligator {

/// Data struct for composite costs.
template <typename Scalar>
struct CompositeCostDataTpl : CostDataAbstractTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostDataAbstractTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;

  shared_ptr<StageFunctionData> residual_data;
  RowMatrixXs JtW_buf;
  VectorXs Wv_buf;
  CompositeCostDataTpl(const int ndx, const int nu,
                       shared_ptr<StageFunctionData> rdata)
      : Base(ndx, nu), residual_data(rdata), JtW_buf(ndx + nu, rdata->nr),
        Wv_buf(rdata->nr) {
    JtW_buf.setZero();
    Wv_buf.setZero();
  }
};

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
                           const MatrixXs &weights);

  void configure(
      CommonModelBuilderContainer &common_buider_container) const override;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data_) const override;

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data_) const override;

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostData &data_) const override;

  shared_ptr<CostData> createData() const override {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData());
  }

  shared_ptr<CostData>
  createData(const CommonModelDataContainer &container) const override {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData(container));
  }

private:
  void debug_dims() const {
    if (residual_->nr != weights_.cols()) {
      ALIGATOR_RUNTIME_ERROR(
          "Weight matrix and residual codimension are inconsistent.");
    }
  }
};

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
                     shared_ptr<StageFunction> function, const VectorXs &scale);

  LogResidualCostTpl(shared_ptr<Manifold> space,
                     shared_ptr<StageFunction> function, const Scalar scale);

  void configure(
      CommonModelBuilderContainer &common_buider_container) const override;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostDataAbstract &data) const override;

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostDataAbstract &data) const override;

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostDataAbstract &data) const override;

  shared_ptr<CostDataAbstract> createData() const override {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData());
  }

  shared_ptr<CostDataAbstract>
  createData(const CommonModelDataContainer &container) const override {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData(container));
  }
};

} // namespace aligator

// Implementation files
#include "aligator/modelling/quad-residual-cost.hxx"
#include "aligator/modelling/log-residual-cost.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./composite-costs.txx"
#endif
