/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/compat/crocoddyl/context.hpp"
#include "aligator/compat/crocoddyl/problem-wrap.hpp"
#include "aligator/compat/crocoddyl/action-model-wrap.hpp"

#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/actions/lqr.hpp>
#include <crocoddyl/core/solvers/ddp.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>

#include "aligator/solvers/proxddp/solver-proxddp.hpp"

#include <boost/test/unit_test.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(crocoddyl_problem)

namespace pcroc = aligator::compat::croc;

BOOST_AUTO_TEST_CASE(lqr) {
  using crocoddyl::ActionModelLQR;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using pcroc::context::ActionDataWrapper;
  using pcroc::context::ActionModelWrapper;

  long nx = 4;
  long nu = 3;
  crocoddyl::StateVector state((std::size_t)nx);
  VectorXd x0 = state.rand();
  VectorXd x1 = state.rand();
  VectorXd u0 = VectorXd::Random(nu);

  MatrixXd luu = MatrixXd::Ones(nu, nu) * 1e-3;
  MatrixXd lxx = MatrixXd::Ones(nx, nx) * 1e-2;
  VectorXd lx0 = -lxx * x1;

  auto lqr_model = std::make_shared<ActionModelLQR>(nx, nu);
  lqr_model->set_Luu(luu);
  lqr_model->set_lx(lx0);
  lqr_model->set_lu(VectorXd::Zero(nu));
  lqr_model->set_Lxu(MatrixXd::Zero(nx, nu));
  auto lqr_data = lqr_model->createData();
  lqr_model->calc(lqr_data, x0, u0);
  lqr_model->calcDiff(lqr_data, x0, u0);

  auto act_wrapper = std::make_shared<ActionModelWrapper>(lqr_model);

  const std::size_t nsteps = 10;

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      nsteps, lqr_model);
  auto croc_problem = std::make_shared<crocoddyl::ShootingProblem>(
      x0, running_models, lqr_model);

  std::vector<VectorXd> xs_init(nsteps + 1, x0);
  std::vector<VectorXd> us_init(nsteps, u0);

  crocoddyl::SolverDDP croc_solver(croc_problem);
  croc_solver.setCallbacks({std::make_shared<crocoddyl::CallbackVerbose>()});
  const double TOL = 1e-8;
  croc_solver.set_th_stop(TOL * TOL);
  bool cr_converged = croc_solver.solve(xs_init, us_init);

  auto croc_xs = croc_solver.get_xs();
  auto croc_us = croc_solver.get_us();

  double cr_cost = croc_solver.get_cost();
  std::size_t cr_iters = croc_solver.get_iter();
  fmt::print("croc #iters: {:d}\n", cr_iters);
  fmt::print("croc cost: {:.3e}\n", cr_cost);

  BOOST_TEST_CHECK(cr_converged);

  // convert to aligator problem

  aligator::TrajOptProblemTpl<double> prox_problem =
      pcroc::convertCrocoddylProblem(croc_problem);

  const double mu_init = 1e-5;
  aligator::SolverProxDDPTpl<double> prox_solver(TOL, mu_init);
  prox_solver.verbose_ = aligator::VerboseLevel::VERBOSE;
  prox_solver.max_iters = 8;
  prox_solver.force_initial_condition_ = true;
  prox_solver.rollout_type_ = aligator::RolloutType::NONLINEAR;

  prox_solver.setup(prox_problem);
  bool conv2 = prox_solver.run(prox_problem, xs_init, us_init);

  const auto &results = prox_solver.results_;
  fmt::print("{}\n", results);
  const auto &xs = results.xs;
  const auto &us = results.us;

  BOOST_TEST_CHECK(conv2);

  for (std::size_t i = 0; i <= nsteps; i++) {
    auto e = aligator::math::infty_norm(xs[i] - croc_xs[i]);
    fmt::print("errx[{:>2d}] = {:.3e}\n", i, e);
    BOOST_CHECK_LE(e, 1e-6);
  }
  for (std::size_t i = 0; i < nsteps; i++) {
    auto e = aligator::math::infty_norm(us[i] - croc_us[i]);
    fmt::print("erru[{:>2d}] = {:.3e}\n", i, e);
    BOOST_CHECK_LE(e, 1e-6);
  }
}

BOOST_AUTO_TEST_SUITE_END()
