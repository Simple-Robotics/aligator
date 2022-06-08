/// A car in SE2


#include <proxnlp/modelling/spaces/pinocchio-groups.hpp>
#include <pinocchio/multibody/liegroup/special-euclidean.hpp>

#include "proxddp/solver-proxddp.hpp"
#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/state-error.hpp"
#include "proxddp/modelling/composite-costs.hpp"


using T = double;
using SE2_t = proxnlp::PinocchioLieGroup<pinocchio::SpecialEuclideanOperationTpl<2, double>>;


using namespace proxddp;

int main()
{
  SE2_t space;
  const int nu = space.ndx();

  const auto x0 = space.rand();
  Eigen::VectorXd u0(nu);
  u0.setRandom();
  const auto x_target = space.neutral();

  auto err_fun = std::make_shared<StateErrorResidual<T>>(space, nu, x_target);
  auto err_fun_data = err_fun->createData();

  err_fun->evaluate(x0, u0, x0, *err_fun_data);
  fmt::print("err fun eval: {}\n", err_fun_data->value_.transpose());
  err_fun->evaluate(x_target, u0, x0, *err_fun_data);
  fmt::print("err fun eval: {}\n", err_fun_data->value_.transpose());


  Eigen::MatrixXd weights(err_fun->nr, err_fun->nr);
  weights.setIdentity();

  auto cost_fun = std::make_shared<QuadResidualCost<T>>(err_fun, weights);
  auto cost_data = cost_fun->createData();
  cost_fun->evaluate(x0, u0, *cost_data);
  fmt::print("cost val(x0)  : {:.3e}\n", cost_data->value_);
  cost_fun->evaluate(x_target, u0, *cost_data);
  fmt::print("cost val(xtar): {:.3e}\n", cost_data->value_);


  return 0;
}

