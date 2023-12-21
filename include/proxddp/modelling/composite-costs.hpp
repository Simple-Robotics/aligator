/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function-abstract.hpp"
#include "proxddp/core/cost-abstract.hpp"
#include "proxddp/modelling/state-error.hpp"

#include <fmt/ostream.h>

namespace aligator {

/// Data struct for composite costs.
template <typename Scalar>
struct CompositeCostDataTpl : CostDataAbstractTpl<Scalar> {
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
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
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  MatrixXs weights_;
  shared_ptr<StageFunction> residual_;
  bool gauss_newton = true;

  QuadraticResidualCostTpl(shared_ptr<Manifold> space,
                           shared_ptr<StageFunction> function,
                           const MatrixXs &weights);

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

private:
  void debug_dims() const {
    if (residual_->nr != weights_.cols()) {
      PROXDDP_RUNTIME_ERROR(
          "Weight matrix and residual codimension are inconsistent.");
    }
  }
};

/// @brief  Log-barrier of an underlying cost function.
template <typename Scalar> struct LogResidualCostTpl : CostAbstractTpl<Scalar> {
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Base = CostAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  VectorXs barrier_weights_;
  shared_ptr<StageFunction> residual_;

  LogResidualCostTpl(shared_ptr<Manifold> space,
                     shared_ptr<StageFunction> function, const VectorXs &scale);

  LogResidualCostTpl(shared_ptr<Manifold> space,
                     shared_ptr<StageFunction> function, const Scalar scale);

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
};

} // namespace aligator

// Implementation files
#include "proxddp/modelling/quad-residual-cost.hxx"
#include "proxddp/modelling/log-residual-cost.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./composite-costs.txx"
#endif
