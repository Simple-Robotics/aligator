/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/compat/crocoddyl/action-model-wrap.hpp"
#include "aligator/compat/crocoddyl/context.hpp"

#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/core/actions/lqr.hpp>

#include <boost/test/unit_test.hpp>

#include <proxsuite-nlp/fmt-eigen.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(croc_action_model)

using namespace aligator;
namespace pcroc = aligator::compat::croc;

BOOST_AUTO_TEST_CASE(lqr) {
  using crocoddyl::ActionModelLQR;
  using pcroc::context::ActionDataWrapper;
  using pcroc::context::ActionModelWrapper;
  long nx = 4;
  long nu = 3;
  crocoddyl::StateVector state((std::size_t)nx);
  Eigen::VectorXd x0 = state.zero();
  Eigen::VectorXd x1 = state.rand();
  Eigen::VectorXd u0(nu);
  u0.setRandom();
  fmt::print("1/2 * |u0|²: {}\n", 0.5 * u0.squaredNorm());

  auto lqr_model = boost::make_shared<ActionModelLQR>(nx, nu);
  lqr_model->set_lx(Eigen::VectorXd::Zero(nx));
  lqr_model->set_lu(Eigen::VectorXd::Zero(nu));
  lqr_model->set_Lxu(Eigen::MatrixXd::Random(nx, nu) * 0.1);
  auto lqr_data = lqr_model->createData();
  lqr_model->calc(lqr_data, x0, u0);
  lqr_model->calcDiff(lqr_data, x0, u0);
  fmt::print("lqr lx_: {}\n", lqr_model->get_lx().transpose());
  fmt::print("lqr lu_: {}\n", lqr_model->get_lu().transpose());
  fmt::print("Cost: {}\n", lqr_data->cost);
  fmt::print("Lx  : {}\n", lqr_data->Lx.transpose());
  fmt::print("Lu  : {}\n", lqr_data->Lu.transpose());

  pcroc::context::ActionModelWrapper act_wrapper(lqr_model);
  auto act_wrap_data = act_wrapper.createData();

  act_wrapper.evaluate(x0, u0, x0, *act_wrap_data);
  act_wrapper.computeFirstOrderDerivatives(x0, u0, x0, *act_wrap_data);

  auto cd = *act_wrap_data->cost_data;
  fmt::print("act cost_data\n");
  fmt::print("cost: {}\n", cd.value_);
  fmt::print("grad: {}\n", cd.grad_.transpose());

  BOOST_TEST_CHECK(cd.value_ == lqr_data->cost);
  BOOST_TEST_CHECK(cd.Lx_.isApprox(lqr_data->Lx));
  BOOST_TEST_CHECK(cd.Lu_.isApprox(lqr_data->Lu));
  BOOST_TEST_CHECK(cd.Lxx_.isApprox(lqr_data->Lxx));
  BOOST_TEST_CHECK(cd.Luu_.isApprox(lqr_data->Luu));

  act_wrapper.evaluate(x0, u0, x1, *act_wrap_data);
}

BOOST_AUTO_TEST_SUITE_END()
