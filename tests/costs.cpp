
#include <boost/test/unit_test.hpp>

#include "aligator/core/function-abstract.hpp"
#include "aligator/modelling/state-error.hpp"

#include "aligator/modelling/costs/quad-state-cost.hpp"
#include <proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp>
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

BOOST_AUTO_TEST_SUITE(costs)

using namespace aligator;
using T = double;
using context::MatrixXs;
using context::VectorXs;
using QuadraticResidualCost = QuadraticResidualCostTpl<T>;

void fd_test(VectorXs x0, VectorXs u0, MatrixXs weights,
             QuadraticResidualCost qres, shared_ptr<context::CostData> data) {

  const xyz::polymorphic<StageFunctionTpl<T>> fun = qres.residual_;
  const auto fd = fun->createData();
  const auto ndx = fd->ndx1;
  const auto nu = fd->nu;
  qres.evaluate(x0, u0, *data);
  qres.computeGradients(x0, u0, *data);
  qres.computeHessians(x0, u0, *data);

  // analytical formula
  fun->evaluate(x0, u0, x0, *fd);
  fun->computeJacobians(x0, u0, x0, *fd);

  auto n = (long)(ndx + nu);
  auto J = fd->jac_buffer_.leftCols(n);

  auto grad_ref = J.transpose() * weights * fd->value_;
  auto hess_ref = J.transpose() * weights * J;
  BOOST_CHECK(grad_ref.isApprox(data->grad_));
  BOOST_CHECK(hess_ref.isApprox(data->hess_));
}

BOOST_AUTO_TEST_CASE(quad_state_se2) {
  using SE2 = proxsuite::nlp::SETpl<2, T>;
  auto space = std::make_shared<SE2>();

  const Eigen::Index ndx = space->ndx();
  const Eigen::Index nu = 1UL;
  Eigen::VectorXd u0(nu);
  u0.setZero();

  const auto target = space->rand();

  const auto fun =
      std::make_shared<StateErrorResidualTpl<T>>(*space, nu, target);

  BOOST_CHECK_EQUAL(fun->nr, ndx);
  Eigen::MatrixXd weights(ndx, ndx);
  weights.setIdentity();
  const auto qres =
      std::make_shared<QuadraticStateCostTpl<T>>(*space, nu, target, weights);

  shared_ptr<context::CostData> data = qres->createData();
  auto fd = fun->createData();

  const int nrepeats = 10;

  for (int k = 0; k < nrepeats; k++) {
    Eigen::VectorXd x0 = space->rand();
    fd_test(x0, u0, weights, *qres, data);
  }
}

BOOST_AUTO_TEST_CASE(quad_state_highdim) {
  using VectorSpace = proxsuite::nlp::VectorSpaceTpl<T>;
  const Eigen::Index ndx = 56;
  const auto space = std::make_shared<VectorSpace>(ndx);
  const Eigen::Index nu = 1UL;

  Eigen::VectorXd u0(nu);
  u0.setZero();

  const auto target = space->rand();

  const auto fun =
      std::make_shared<StateErrorResidualTpl<T>>(*space, nu, target);

  BOOST_CHECK_EQUAL(fun->nr, ndx);
  Eigen::MatrixXd weights(ndx, ndx);
  weights.setIdentity();
  const auto qres =
      std::make_shared<QuadraticStateCostTpl<T>>(*space, nu, target, weights);

  shared_ptr<context::CostData> data = qres->createData();
  auto fd = fun->createData();

  const int nrepeats = 10;

  for (int k = 0; k < nrepeats; k++) {
    Eigen::VectorXd x0 = space->rand();
    fd_test(x0, u0, weights, *qres, data);
  }
}

BOOST_AUTO_TEST_SUITE_END()
