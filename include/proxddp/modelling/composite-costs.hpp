#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function.hpp"
#include "proxddp/core/costs.hpp"
#include "proxddp/modelling/state-error.hpp"


  namespace proxddp
{
  /// Data struct for composite costs.
  template<typename Scalar>
  struct CompositeCostDataTpl : CostDataAbstract<Scalar>
  {
    shared_ptr<FunctionDataTpl<Scalar>> underlying_data;
    CompositeCostDataTpl(const int ndx, const int nu)
      : CostDataAbstract<Scalar>(ndx, nu)
      {}
  };

  /** @brief Quadratic composite of an underlying function.
   *
   * @details This is defined as
   * \f[
   *      c(x, u) \overset{\triangle}{=} \frac{1}{2} \|r(x, u)\|_W^2.
   * \f]
   */
  template<typename _Scalar>
  struct QuadraticResidualCost : CostAbstractTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using CostData = CostDataAbstract<Scalar>;
    using CompositeData = CompositeCostDataTpl<Scalar>;

    MatrixXs weights_;
    shared_ptr<const StageFunctionTpl<Scalar>> residual_;
    bool gauss_newton = true;

    QuadraticResidualCost(const shared_ptr<StageFunctionTpl<Scalar>>& function,
                     const MatrixXs& weights)
      : CostAbstractTpl<Scalar>(function->ndx1, function->nu)
      , weights_(weights)
      , residual_(function)
      {
        assert(residual_->nr == weights.cols());
      }

    void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data_) const
    {
      auto& data = static_cast<CompositeData&>(data_);
      auto& under_data = *data.underlying_data;
      residual_->evaluate(x, u, x, under_data);
      data.value_ = .5 * under_data.value_.dot(weights_ * under_data.value_);
    }

    void computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data_) const
    {
      auto& data = static_cast<CompositeData&>(data_);
      auto& under_data = *data.underlying_data;
      residual_->computeJacobians(x, u, x, under_data);
      data.grad_ = under_data.jac_buffer_.transpose() * (weights_ * under_data.value_);
    }

    void computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data_) const
    {
      auto& data = static_cast<CompositeData&>(data_);
      auto& under_data = *data.underlying_data;
      data.hess_ = under_data.jac_buffer_.transpose() * (weights_ * under_data.jac_buffer_);
      if (!gauss_newton)
      {
        residual_->computeVectorHessianProducts(x, u, x, weights_ * under_data.value_, under_data);
        data.hess_.noalias() += under_data.vhp_buffer_;
      }
    }

    shared_ptr<CostData> createData() const
    {
      CompositeData* d = new CompositeData{this->ndx_, this->nu_};
      d->underlying_data = std::move(residual_->createData());
      return shared_ptr<CostData>(std::move(d));
    }

  };

  /// Log-barrier of an underlying cost function.
  template<typename Scalar>
  struct LogResidualCost : CostAbstractTpl<Scalar>
  {
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using CostData = CostDataAbstract<Scalar>;
    using CompositeData = CompositeCostDataTpl<Scalar>;
    using StageFunction = StageFunctionTpl<Scalar>;

    VectorXs barrier_weights_;
    shared_ptr<StageFunction> residual_;

    LogResidualCost(const shared_ptr<StageFunction>& function,
                    const VectorXs scale)
      : residual_(function)
      , barrier_weights_(scale) {
      assert(scale.size() == function->nr);
    }

    LogResidualCost(const shared_ptr<StageFunction>& function,
                    const Scalar scale)
      : LogResidualCost(function, VectorXs::Constant(function->nr, scale))
      {}

    void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {
      auto& d = static_cast<CompositeData&>(data);
      residual_->evaluate(x, u, x, *d.underlying_data);
      d.value_ = barrier_weights_.dot(d.underlying_data->value_.log());
    }

    void computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {
      auto& d = static_cast<CompositeData&>(data);
      auto& under_d = *d.underlying_data;
      residual_->computeJacobians(x, u, x, under_d);
      d.grad_.setZero();
      VectorXs& v = under_d.value_;
      const int nrows = residual_->nr;
      for (int i = 0; i < nrows; i++)
      {
        d.grad_ += barrier_weights_(i) * under_d.jac_buffer_.row(i) / v(i);
      }
    }

    void computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {
      auto& d = static_cast<CompositeData&>(data);
      auto& under_d = *d.underlying_data;
      d.hess_.setZero();
      VectorXs& v = under_d.value_;
      const int nrows = residual_->nr;
      for (int i = 0; i < nrows; i++)
      {
        VectorRef g_i = under_d.jac_buffer_.row(i);
        d.hess_ += barrier_weights_(i) * (g_i * g_i.transpose()) / (v(i) * v(i));
      }
    }
  };

  template<typename Scalar>
  shared_ptr<QuadraticResidualCost<Scalar>>
  make_state_distance_cost(
    const typename math_types<Scalar>::MatrixXs& weights,
    const ManifoldAbstractTpl<Scalar>& space,
    const int nu,
    const typename math_types<Scalar>::VectorXs& target)
  {
    return std::make_shared<QuadraticResidualCost<Scalar>>(
      std::make_shared<StateErrorResidual<Scalar>>(space, nu, target),
      weights);
  }

} // namespace proxddp
