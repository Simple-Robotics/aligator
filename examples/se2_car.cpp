/// A car in SE2

#include <proxnlp/modelling/spaces/pinocchio-groups.hpp>
#include <pinocchio/multibody/liegroup/special-euclidean.hpp>

#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/state-error.hpp"
#include "proxddp/modelling/composite-costs.hpp"

using T = double;
using SE2_t =
    proxnlp::PinocchioLieGroup<pinocchio::SpecialEuclideanOperationTpl<2, T>>;

using namespace proxddp;

int main() {
  auto spaceptr = std::make_shared<SE2_t>();
  const auto &space = *spaceptr;
  const int nu = space.ndx();

  const Eigen::VectorXd x0 = space.rand();
  Eigen::VectorXd u0(nu);
  u0.setRandom();
  const Eigen::VectorXd x_target = space.neutral();

  auto err_fun =
      std::make_shared<StateErrorResidual<T>>(spaceptr, nu, x_target);
  auto err_fun_data = err_fun->createData();

  err_fun->evaluate(x0, u0, x0, *err_fun_data);
  fmt::print("err fun eval: {}\n", err_fun_data->value_.transpose());
  err_fun->evaluate(x_target, u0, x0, *err_fun_data);
  fmt::print("err fun eval: {}\n", err_fun_data->value_.transpose());

  Eigen::MatrixXd weights(err_fun->nr, err_fun->nr);
  weights.setIdentity();

  shared_ptr<QuadraticResidualCost<T>> cost_fun =
      std::make_shared<QuadraticResidualCost<T>>(err_fun, weights);
  auto cd1 = cost_fun->createData();
  cost_fun->evaluate(x0, u0, *cd1);
  fmt::print("cost val(x0)  : {:.3e}\n", cd1->value_);
  cost_fun->evaluate(x_target, u0, *cd1);
  fmt::print("cost val(xtar): {:.3e}\n", cd1->value_);

  return 0;
}
