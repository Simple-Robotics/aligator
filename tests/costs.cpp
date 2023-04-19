
#include <boost/test/unit_test.hpp>

#include "proxddp/modelling/state-error.hpp"
#include "proxddp/modelling/composite-costs.hpp"

#include <proxnlp/modelling/spaces/pinocchio-groups.hpp>

BOOST_AUTO_TEST_SUITE(costs)

using namespace proxddp;
using T = double;

BOOST_AUTO_TEST_CASE(quad_state) {
  using SE2 = proxnlp::SETpl<2, T>;
  auto space = std::make_shared<SE2>();

  std::size_t ndx = (std::size_t)space->ndx();
  std::size_t nu = 1UL;
  Eigen::VectorXd u0(nu);
  u0.setZero();

  auto target = space->rand();

  auto fun = std::make_shared<StateErrorResidualTpl<T>>(space, nu, target);

  Eigen::MatrixXd weights(fun->nr, fun->nr);
  weights.setIdentity();
  auto qres =
      std::make_shared<QuadraticResidualCostTpl<T>>(space, fun, weights);

  shared_ptr<CostDataAbstractTpl<T>> data = qres->createData();
  auto fd = fun->createData();

  int nrepeats = 10;

  int k = 0;
  while (k < nrepeats) {
    Eigen::VectorXd x0 = space->rand();
    qres->evaluate(x0, u0, *data);
    qres->computeGradients(x0, u0, *data);
    qres->computeHessians(x0, u0, *data);

    fmt::print("grad: {}\n", data->grad_.transpose());
    fmt::print("hess:\n{}\n", data->hess_);

    // analytical formula
    fun->evaluate(x0, u0, x0, *fd);
    fun->computeJacobians(x0, u0, x0, *fd);

    auto n = (long)(ndx + nu);
    auto J = fd->jac_buffer_.leftCols(n);

    auto grad_ref = J.transpose() * weights * fd->value_;
    auto hess_ref = J.transpose() * weights * J;
    fmt::print("grad_ref: {}\n", grad_ref.transpose());
    fmt::print("hess_ref:\n{}\n", hess_ref);
    BOOST_CHECK(grad_ref.isApprox(data->grad_));
    BOOST_CHECK(hess_ref.isApprox(data->hess_));

    k++;
  }
}

BOOST_AUTO_TEST_SUITE_END()
