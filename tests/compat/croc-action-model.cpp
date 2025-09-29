/// @copyright Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA
#include "aligator/compat/crocoddyl/action-model-wrap.hpp"
#include "aligator/compat/crocoddyl/context.hpp"

#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/core/actions/lqr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <aligator/fmt-eigen.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace aligator;
namespace pcroc = aligator::compat::croc;

TEST_CASE("lqr") {
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

  auto lqr_model = std::make_shared<ActionModelLQR>(nx, nu);
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

  REQUIRE(cd.value_ == lqr_data->cost);
  REQUIRE(cd.Lx_.isApprox(lqr_data->Lx));
  REQUIRE(cd.Lu_.isApprox(lqr_data->Lu));
  REQUIRE(cd.Lxx_.isApprox(lqr_data->Lxx));
  REQUIRE(cd.Luu_.isApprox(lqr_data->Luu));

  act_wrapper.evaluate(x0, u0, x1, *act_wrap_data);
}
