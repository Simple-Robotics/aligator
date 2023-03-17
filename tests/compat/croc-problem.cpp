/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/compat/crocoddyl/context.hpp"
#include "proxddp/compat/crocoddyl/problem-wrap.hpp"

#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/actions/lqr.hpp>
#include <crocoddyl/core/solvers/ddp.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>

#include "proxddp/core/solver-proxddp.hpp"

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(crocoddyl_problem)

namespace pcroc = proxddp::compat::croc;

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

  auto lqr_model = boost::make_shared<ActionModelLQR>(nx, nu);
  lqr_model->set_Luu(luu);
  lqr_model->set_lx(lx0);
  lqr_model->set_lu(VectorXd::Zero(nu));
  lqr_model->set_Lxu(MatrixXd::Zero(nx, nu));
  auto lqr_data = lqr_model->createData();
  lqr_model->calc(lqr_data, x0, u0);
  lqr_model->calcDiff(lqr_data, x0, u0);

  auto act_wrapper = std::make_shared<ActionModelWrapper>(lqr_model);

  const std::size_t nsteps = 10;

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      nsteps, lqr_model);
  auto croc_problem = boost::make_shared<crocoddyl::ShootingProblem>(
      x0, running_models, lqr_model);

  crocoddyl::SolverDDP croc_solver(croc_problem);
  croc_solver.setCallbacks({boost::make_shared<crocoddyl::CallbackVerbose>()});
  const double TOL = 1e-7;
  croc_solver.set_th_stop(TOL * TOL);
  bool cr_converged = croc_solver.solve();

  auto croc_xs = croc_solver.get_xs();
  auto croc_us = croc_solver.get_us();

  double cr_cost = croc_solver.get_cost();
  std::size_t cr_iters = croc_solver.get_iter();
  fmt::print("croc #iters: {:d}\n", cr_iters);
  fmt::print("croc cost: {:.3e}\n", cr_cost);

  BOOST_TEST_CHECK(cr_converged);

  // convert to proxddp problem

  proxddp::TrajOptProblemTpl<double> prox_problem =
      pcroc::convertCrocoddylProblem(croc_problem);

  const double mu_init = 1e-4;
  proxddp::SolverProxDDP<double> prox_solver(TOL, mu_init);
  prox_solver.verbose_ = proxddp::VerboseLevel::VERBOSE;
  prox_solver.max_iters = 8;
#ifndef NDEBUG
  prox_solver.dump_linesearch_plot = true;
#endif
  prox_solver.rollout_type_ = proxddp::RolloutType::NONLINEAR;

  std::vector<VectorXd> xs_init(nsteps + 1, x0);
  std::vector<VectorXd> us_init(nsteps, u0);

  prox_solver.setup(prox_problem);
  bool conv2 = prox_solver.run(prox_problem, xs_init, us_init);

  const auto &results = prox_solver.results_;
  fmt::print("{}\n", results);

  BOOST_TEST_CHECK(conv2);

  for (std::size_t i = 0; i < nsteps; i++) {
    BOOST_TEST_CHECK(results.xs[i].isApprox(croc_xs[i], 1e-4));
  }
}

BOOST_AUTO_TEST_SUITE_END()
