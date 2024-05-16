/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./quad-residual-cost.hpp"

namespace aligator {

template <typename Scalar>
QuadraticResidualCostTpl<Scalar>::QuadraticResidualCostTpl(
    shared_ptr<Manifold> space, shared_ptr<StageFunction> function,
    const ConstMatrixRef &weights)
    : Base(space, function->nu), weights_(weights), residual_(function) {
  if (residual_->nr != weights_.cols()) {
    ALIGATOR_RUNTIME_ERROR(
        "Weight matrix and residual codimension are inconsistent.");
  }
}

template <typename Scalar>
void QuadraticResidualCostTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                const ConstVectorRef &u,
                                                CostData &data_) const {
  Data &data = static_cast<Data &>(data_);
  StageFunctionDataTpl<Scalar> &under_data = *data.residual_data;
  residual_->evaluate(x, u, x, under_data);
  ALIGATOR_NOMALLOC_SCOPED;
  data.Wv_buf.noalias() = weights_ * under_data.value_;
  data.value_ = .5 * under_data.value_.dot(data.Wv_buf);
}

template <typename Scalar>
void QuadraticResidualCostTpl<Scalar>::computeGradients(const ConstVectorRef &x,
                                                        const ConstVectorRef &u,
                                                        CostData &data_) const {
  Data &data = static_cast<Data &>(data_);
  StageFunctionDataTpl<Scalar> &under_data = *data.residual_data;
  residual_->computeJacobians(x, u, x, under_data);
  const Eigen::Index size = data.grad_.size();
  ALIGATOR_NOMALLOC_SCOPED;
  MatrixRef J = under_data.jac_buffer_.leftCols(size);
  data.Wv_buf.noalias() = weights_ * under_data.value_;
  data.grad_.noalias() = J.transpose() * data.Wv_buf;
}

template <typename Scalar>
void QuadraticResidualCostTpl<Scalar>::computeHessians(const ConstVectorRef &x,
                                                       const ConstVectorRef &u,
                                                       CostData &data_) const {
  Data &data = static_cast<Data &>(data_);
  StageFunctionDataTpl<Scalar> &under_data = *data.residual_data;
  const Eigen::Index size = data.grad_.size();
  MatrixRef J = under_data.jac_buffer_.leftCols(size);
  data.JtW_buf.noalias() = J.transpose() * weights_;
  data.hess_ = data.JtW_buf * J;
  if (!gauss_newton) {
    residual_->computeVectorHessianProducts(x, u, x, data.Wv_buf, under_data);
    data.hess_ = under_data.vhp_buffer_;
  }
}

} // namespace aligator
