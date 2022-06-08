/// A car in SE2


#include <proxnlp/modelling/spaces/pinocchio-groups.hpp>
#include <pinocchio/multibody/liegroup/special-euclidean.hpp>

#include "proxddp/solver-proxddp.hpp"
#include "proxddp/modelling/costs.hpp"
#include "proxddp/core/explicit-dynamics.hpp"


using T = double;
using SE2_t = proxnlp::PinocchioLieGroup<pinocchio::SpecialEuclideanOperationTpl<2, double>>;


namespace proxddp
{

  /// @brief \f$ x \ominus x_{tar} \f$
  struct QuadErrState : StageFunctionTpl<T>
  {
    PROXNLP_DYNAMIC_TYPEDEFS(T);

    VectorXs target;
    const ManifoldAbstractTpl<T>& space;

    QuadErrState(const ManifoldAbstractTpl<T>& space,
                const int nu,
                const VectorXs& target)
      : StageFunctionTpl<T>(space.ndx(), nu, space.ndx())
      , target(target), space(space) {}

    void evaluate(const ConstVectorRef& x,
                  const ConstVectorRef& u,
                  const ConstVectorRef& y,
                  Data& data) const
    {
      space.difference(x, target, data.value_);
    }

    void computeJacobians(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& y,
                          Data& data) const
    {
      space.Jdifference(x, target, data.Jx_, 0);
    }
  };

  struct QuadCostData : CostDataAbstract<T>
  {
    QuadCostData(const int ndx, const int nu) : CostDataAbstract<T>(ndx, nu) {}
    shared_ptr<FunctionDataTpl<T>> underlying_data;
  };

  struct QuadResidualCost : CostBaseTpl<T>
  {
    PROXNLP_DYNAMIC_TYPEDEFS(T);

    MatrixXs weights_;
    shared_ptr<StageFunctionTpl<T>> residual_;
    bool gauss_newton = true;

    QuadResidualCost(const shared_ptr<StageFunctionTpl<T>>& function,
                    const MatrixXs& weights)
      : CostBaseTpl<T>(function->ndx1, function->nu)
      , weights_(weights)
      , residual_(function)
      {
        assert(residual_->nr == weights.cols());
      }

    void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data_) const
    {
      auto data = static_cast<QuadCostData&>(data_);
      auto& under_data = *data.underlying_data;
      residual_->evaluate(x, u, x, under_data);
      data.value_ = .5 * under_data.value_.dot(weights_ * under_data.value_);
      fmt::print("computed cost: {:.3e}\n", data.value_);
    }

    void computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data_) const
    {
      auto data = static_cast<QuadCostData&>(data_);
      auto& under_data = *data.underlying_data;
      residual_->computeJacobians(x, u, x, under_data);
      data.grad_ = (weights_ * under_data.jac_buffer_.transpose()) * under_data.value_;
    }

    void computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data_) const
    {
      auto data = static_cast<QuadCostData&>(data_);
      auto& under_data = *data.underlying_data;
      data.hess_ = under_data.jac_buffer_.transpose() * (weights_ * under_data.jac_buffer_);
      if (!gauss_newton)
      {
        residual_->computeVectorHessianProducts(x, u, x, weights_ * under_data.value_, under_data);
        data.hess_.noalias() += under_data.vhp_buffer_;
      }
    }

    shared_ptr<CostDataAbstract<T>> createData() const
    {
      auto d = new QuadCostData{this->ndx_, this->nu_};
      d->underlying_data = residual_->createData();
      return shared_ptr<CostDataAbstract<T>>(d);
    }

  };
    
} // namespace proxddp


using namespace proxddp;

int main()
{
  SE2_t space;
  int nu = space.ndx();

  auto x0 = space.rand();
  Eigen::VectorXd u0(nu);
  u0.setRandom();
  auto x_target = space.neutral();

  auto err_fun = std::make_shared<QuadErrState>(space, nu, x_target);
  auto qd_fun_data = err_fun->createData();

  err_fun->evaluate(x0, u0, x0, *qd_fun_data);
  fmt::print("err fun eval: {}\n", qd_fun_data->value_.transpose());
  err_fun->evaluate(x_target, u0, x0, *qd_fun_data);
  fmt::print("err fun eval: {}\n", qd_fun_data->value_.transpose());


  Eigen::MatrixXd weights(err_fun->nr, err_fun->nr);
  weights.setIdentity();

  auto new_cost = std::make_shared<QuadResidualCost>(err_fun, weights);
  auto cost_data = new_cost->createData();
  new_cost->evaluate(x0, u0, *cost_data);
  fmt::print("cost val(x0)  : {:.3e}\n", cost_data->value_);
  new_cost->evaluate(x_target, u0, *cost_data);
  fmt::print("cost val(xtar): {:.3e}\n", cost_data->value_);


  return 0;
}

