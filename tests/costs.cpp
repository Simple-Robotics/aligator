
#include <boost/test/unit_test.hpp>

#include "aligator/core/function-abstract.hpp"
#include "aligator/modelling/state-error.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"

#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/spaces/pinocchio-groups.hpp"
#include "aligator/core/vector-space.hpp"

BOOST_AUTO_TEST_SUITE(costs)

using namespace aligator;
using T = double;
using context::CostData;
using context::MatrixXs;
using context::VectorXs;
using QuadraticResidualCost = QuadraticResidualCostTpl<T>;
using StateError = StateErrorResidualTpl<T>;
using SE2 = SETpl<2, T>;
using VectorSpace = VectorSpaceTpl<T>;

void fd_test(VectorXs x0, VectorXs u0, MatrixXs weights,
             QuadraticResidualCost qres, shared_ptr<CostData> data) {

  const xyz::polymorphic<StageFunctionTpl<T>> fun = qres.residual_;
  const auto fd = fun->createData();
  const auto ndx = fd->ndx1;
  const auto nu = fd->nu;
  qres.evaluate(x0, u0, *data);
  qres.computeGradients(x0, u0, *data);
  qres.computeHessians(x0, u0, *data);

  // analytical formula
  fun->evaluate(x0, u0, *fd);
  fun->computeJacobians(x0, u0, *fd);

  auto n = (long)(ndx + nu);
  auto J = fd->jac_buffer_.leftCols(n);

  auto grad_ref = J.transpose() * weights * fd->value_;
  auto hess_ref = J.transpose() * weights * J;
  BOOST_CHECK(grad_ref.isApprox(data->grad_));
  BOOST_CHECK(hess_ref.isApprox(data->hess_));
}

BOOST_AUTO_TEST_CASE(quad_state_se2) {
  SE2 space;

  const Eigen::Index ndx = space.ndx();
  const Eigen::Index nu = 1UL;
  Eigen::VectorXd u0(nu);
  u0.setZero();

  const auto target = space.rand();

  const StateError fun(space, nu, target);

  BOOST_CHECK_EQUAL(fun.nr, ndx);
  Eigen::MatrixXd weights(ndx, ndx);
  weights.setIdentity();
  const QuadraticStateCostTpl<T> qres(space, nu, target, weights);

  shared_ptr<CostData> data = qres.createData();
  auto fd = fun.createData();

  const int nrepeats = 10;

  for (int k = 0; k < nrepeats; k++) {
    Eigen::VectorXd x0 = space.rand();
    fd_test(x0, u0, weights, qres, data);
  }

  const StateError *fun_cast = qres.getResidual<StateError>();
  BOOST_CHECK(fun_cast != nullptr);
}

BOOST_AUTO_TEST_CASE(quad_state_highdim) {
  const Eigen::Index ndx = 56;
  const VectorSpace space(ndx);
  const Eigen::Index nu = 1UL;

  Eigen::VectorXd u0(nu);
  u0.setZero();

  const auto target = space.rand();

  const StateError fun(space, nu, target);

  BOOST_CHECK_EQUAL(fun.nr, ndx);
  Eigen::MatrixXd weights(ndx, ndx);
  weights.setIdentity();
  const QuadraticStateCostTpl<T> qres(space, nu, target, weights);

  shared_ptr<CostData> data = qres.createData();
  auto fd = fun.createData();

  const int nrepeats = 10;

  for (int k = 0; k < nrepeats; k++) {
    Eigen::VectorXd x0 = space.rand();
    fd_test(x0, u0, weights, qres, data);
  }

  const StateError *fun_cast = qres.getResidual<StateError>();
  BOOST_CHECK(fun_cast != nullptr);

  {
    const auto *try_cast = qres.getResidual<ControlErrorResidualTpl<T>>();
    BOOST_CHECK(try_cast == nullptr);
  }
}

BOOST_AUTO_TEST_CASE(cost_stack) {
  using CostStack = CostStackTpl<T>;
  SE2 space;
  auto nu = space.ndx();
  StateError err{space, nu, space.rand()};
  QuadraticResidualCost qres{space, err,
                             MatrixXs::Identity(err.ndx1, err.ndx1)};
  CostStack cost{space, nu};
  cost.addCost("state", qres, 1.);
  {
    QuadraticResidualCost *pres =
        cost.getComponent<QuadraticResidualCost>("state");
    BOOST_CHECK(pres != nullptr);

    StateError *perr = pres->getResidual<StateError>();
    BOOST_CHECK(perr != nullptr);
  }
}

BOOST_AUTO_TEST_SUITE_END()
