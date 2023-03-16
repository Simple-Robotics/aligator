/// A car in SE2

#include <proxnlp/modelling/spaces/pinocchio-groups.hpp>
#include <pinocchio/multibody/liegroup/special-euclidean.hpp>

#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/state-error.hpp"
#include "proxddp/modelling/composite-costs.hpp"
#include "proxddp/modelling/sum-of-costs.hpp"

using T = double;
using pinocchio::SpecialEuclideanOperationTpl;
using proxnlp::PinocchioLieGroup;
using SE2 = PinocchioLieGroup<SpecialEuclideanOperationTpl<2, T>>;

using namespace proxddp;
using StateError = StateErrorResidualTpl<T>;
using QuadResidualCost = QuadraticResidualCostTpl<T>;

int main() {
  shared_ptr<SE2> spaceptr = std::make_shared<SE2>();
  const SE2 &space = *spaceptr;
  const int nu = space.ndx();

  const Eigen::VectorXd x0 = space.rand();
  Eigen::VectorXd u0(nu);
  u0.setRandom();
  const Eigen::VectorXd x_target = space.neutral();

  auto state_err = std::make_shared<StateError>(spaceptr, nu, x_target);
  /* test */
  {
    auto state_err_data = state_err->createData();

    state_err->evaluate(x0, u0, x0, *state_err_data);
    fmt::print("err fun eval: {}\n", state_err_data->value_.transpose());
    state_err->evaluate(x_target, u0, x0, *state_err_data);
    fmt::print("err fun eval: {}\n", state_err_data->value_.transpose());
  }

  const int nr = state_err->nr;

  Eigen::MatrixXd w_x(nr, nr);
  w_x.diagonal().array() = 0.01;
  Eigen::MatrixXd w_term = w_x * 10;
  Eigen::MatrixXd w_u = w_x;

  const T dt = 0.01;

  auto rcost = std::make_shared<CostStackTpl<T>>(space.ndx(), nu);
  auto rc1 = std::make_shared<QuadResidualCost>(state_err, w_x * dt);
  auto rc2 = std::make_shared<QuadResidualCost>(
      std::make_shared<ControlErrorResidualTpl<T>>(space.ndx(), nu), w_u * dt);
  rcost->addCost(rc1);
  rcost->addCost(rc2);

  {
    auto cd1 = rcost->createData();
    rcost->evaluate(x0, u0, *cd1);
    fmt::print("cost val(x0)  : {:.3e}\n", cd1->value_);
    rcost->evaluate(x_target, u0, *cd1);
    fmt::print("cost val(xtar): {:.3e}\n", cd1->value_);
  }

  auto term_cost = std::make_shared<QuadResidualCost>(state_err, w_term);

  return 0;
}
