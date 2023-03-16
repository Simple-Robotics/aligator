#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function-abstract.hpp"
#include "proxddp/core/cost-abstract.hpp"
#include "proxddp/modelling/state-error.hpp"

#include <fmt/ostream.h>

namespace proxddp {

/// Data struct for composite costs.
template <typename Scalar>
struct CompositeCostDataTpl : CostDataAbstractTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostDataAbstractTpl<Scalar>;
  using FunctionData = FunctionDataTpl<Scalar>;
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;

  shared_ptr<FunctionData> residual_data;
  RowMatrixXs JtW_buf;
  VectorXs Wv_buf;
  CompositeCostDataTpl(const int ndx, const int nu,
                       shared_ptr<FunctionData> rdata)
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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;

  MatrixXs weights_;
  shared_ptr<StageFunctionTpl<Scalar>> residual_;
  bool gauss_newton = true;

  QuadraticResidualCostTpl(shared_ptr<StageFunctionTpl<Scalar>> function,
                           const MatrixXs &weights)
      : CostAbstractTpl<Scalar>(function->ndx1, function->nu),
        weights_(weights), residual_(function) {
    debug_dims();
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostDataAbstract &data_) const {
    Data &data = static_cast<Data &>(data_);
    FunctionDataTpl<Scalar> &under_data = *data.residual_data;
    residual_->evaluate(x, u, x, under_data);
    data.value_ = .5 * under_data.value_.dot(weights_ * under_data.value_);
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostDataAbstract &data_) const {
    Data &data = static_cast<Data &>(data_);
    FunctionDataTpl<Scalar> &under_data = *data.residual_data;
    residual_->computeJacobians(x, u, x, under_data);
    const Eigen::Index size = data.grad_.size();
    MatrixRef J = under_data.jac_buffer_.leftCols(size);
    data.Wv_buf.noalias() = weights_ * under_data.value_;
    data.grad_.noalias() = J.transpose() * data.Wv_buf;
  }

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostDataAbstract &data_) const {
    Data &data = static_cast<Data &>(data_);
    FunctionDataTpl<Scalar> &under_data = *data.residual_data;
    const Eigen::Index size = data.grad_.size();
    MatrixRef J = under_data.jac_buffer_.leftCols(size);
    data.JtW_buf.noalias() = J.transpose() * weights_;
    data.hess_ = data.JtW_buf * J;
    if (!gauss_newton) {
      residual_->computeVectorHessianProducts(x, u, x, data.Wv_buf, under_data);
      data.hess_ = under_data.vhp_buffer_;
    }
  }

  shared_ptr<CostDataAbstract> createData() const {
    return std::make_shared<Data>(this->ndx, this->nu, residual_->createData());
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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Base = CostAbstractTpl<Scalar>;

  VectorXs barrier_weights_;
  shared_ptr<StageFunction> residual_;

  LogResidualCostTpl(shared_ptr<StageFunction> function, const VectorXs &scale)
      : Base(function->ndx1, function->nu), barrier_weights_(scale),
        residual_(function) {
    if (scale.size() != function->nr) {
      PROXDDP_RUNTIME_ERROR(fmt::format(
          "scale argument dimension ({:d}) != function codimension ({:d})",
          scale.size(), function->nr));
    }
    bool negs = (scale.array() <= 0.0).any();
    if (negs) {
      PROXDDP_RUNTIME_ERROR("scale coefficients must be > 0.");
    }
  }

  LogResidualCostTpl(shared_ptr<StageFunction> function, const Scalar scale)
      : LogResidualCostTpl(function, VectorXs::Constant(function->nr, scale)) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostDataAbstract &data) const {
    Data &d = static_cast<Data &>(data);
    residual_->evaluate(x, u, x, *d.residual_data);
    d.value_ =
        barrier_weights_.dot(d.residual_data->value_.array().log().matrix());
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostDataAbstract &data) const {
    Data &d = static_cast<Data &>(data);
    FunctionDataTpl<Scalar> &under_d = *d.residual_data;
    residual_->computeJacobians(x, u, x, under_d);
    d.grad_.setZero();
    VectorXs &v = under_d.value_;
    const int nrows = residual_->nr;
    for (int i = 0; i < nrows; i++) {
      auto g_i = under_d.jac_buffer_.row(i);
      d.grad_.noalias() += barrier_weights_(i) * g_i / v(i);
    }
  }

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostDataAbstract &data) const {
    Data &d = static_cast<Data &>(data);
    FunctionDataTpl<Scalar> &under_d = *d.residual_data;
    d.hess_.setZero();
    VectorXs &v = under_d.value_;
    const int nrows = residual_->nr;
    for (int i = 0; i < nrows; i++) {
      auto g_i = under_d.jac_buffer_.row(i); // row vector
      d.hess_.noalias() +=
          barrier_weights_(i) * (g_i.transpose() * g_i) / (v(i) * v(i));
    }
  }

  shared_ptr<CostDataAbstract> createData() const {
    return std::make_shared<Data>(this->ndx, this->nu, residual_->createData());
  }
};

} // namespace proxddp
