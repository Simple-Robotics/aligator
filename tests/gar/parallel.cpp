/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

#include "./test_util.hpp"
#include "aligator/gar/parallel-solver.hpp"

using namespace aligator::gar;

BOOST_AUTO_TEST_CASE(parallel) {
  uint nx = 1;
  uint nu = 1;
  VectorXs x0;
  x0.setRandom(nx);
  uint horizon = 12;

  problem_t problem = generate_problem(x0, horizon, nx, nu);
  prox_riccati_t solver_full_horz(problem);

  const double mu = 1e-14;
  solver_full_horz.backward(mu, mu);

  constexpr double EPS = 1e-9;

  auto _sol_full = lqrInitializeSolution(problem);
  {
    auto &xs = _sol_full[0];
    auto &us = _sol_full[1];
    auto &vs = _sol_full[2];
    auto &lbdas = _sol_full[3];
    bool ret = solver_full_horz.forward(xs, us, vs, lbdas);
    BOOST_CHECK(ret);
    KktError err_full = compute_kkt_error(problem, xs, us, vs, lbdas);
    printError(err_full, "KKT error (full horz.)");
    BOOST_CHECK_LE(err_full.max, EPS);
  }

  fmt::print("=== solve parallel: ===\n");
  uint t0 = horizon / 2;
  auto prs = splitProblemInTwo(problem, t0, mu);
  auto &pr1 = prs[0];
  auto &pr2 = prs[1];

  BOOST_CHECK_EQUAL(pr1.horizon() + pr2.horizon() + 1, horizon);

  prox_riccati_t subSolve1(pr1);
  prox_riccati_t subSolve2(pr2);

  auto _sol1 = lqrInitializeSolution(pr1);
  auto _sol2 = lqrInitializeSolution(pr2);
  MatrixXs thHess = MatrixXs::Zero(nx, nx);
  VectorXs thGrad = VectorXs::Zero(nx);
  VectorXs thtopt = VectorXs::Zero(nx);
  Eigen::LDLT<MatrixXs> hessChol(nx);

  fmt::print("Available threads: {:d}\n",
             aligator::omp::get_available_threads());

#pragma omp parallel sections num_threads(2)
  // #pragma omp sections
  {
    fmt::print("Current threads: {:d}\n", aligator::omp::get_current_threads());

#pragma omp section
    subSolve1.backward(mu, mu);
#pragma omp section
    subSolve2.backward(mu, mu);
  }
  {
    thHess = subSolve1.thHess + subSolve2.thHess;
    thGrad = subSolve1.thGrad + subSolve2.thGrad;
    hessChol.compute(thHess);
    thtopt = hessChol.solve(-thGrad);
    fmt::print("Theta system err = {:.3e}\n",
               infty_norm(thHess * thtopt + thGrad));
  }
  {
    // subSolve1.forward(_sol1[0], _sol1[1], _sol1[2], _sol1[3]);
    subSolve1.forward(_sol1[0], _sol1[1], _sol1[2], _sol1[3],
                      ConstVectorRef(thtopt));
    // subSolve2.forward(_sol2[0], _sol2[1], _sol2[2], _sol2[3]);
    subSolve2.forward(_sol2[0], _sol2[1], _sol2[2], _sol2[3],
                      ConstVectorRef(thtopt));
  }
  auto &xs1 = _sol1[0];
  auto &us1 = _sol1[1];
  auto &vs1 = _sol1[2];
  auto &lbdas1 = _sol1[3];
  auto &xs2 = _sol2[0];
  auto &us2 = _sol2[1];
  auto &vs2 = _sol2[2];
  auto &lbdas2 = _sol2[3];

  KktError err1 = compute_kkt_error(pr1, xs1, us1, vs1, lbdas1, thtopt);
  KktError err2 = compute_kkt_error(pr2, xs2, us2, vs2, lbdas2, thtopt);

  BOOST_CHECK_LE(err1.max, EPS);
  BOOST_CHECK_LE(err2.max, EPS);

  fmt::print("err1 = {:.3e}\n", err1.max);
  fmt::print("err2 = {:.3e}\n", err2.max);

  uint t1 = horizon - t0;

  VectorOfVectors xs_merge = mergeStdVectors(xs1, xs2);
  VectorOfVectors us_merge = mergeStdVectors(us1, us2);
  VectorOfVectors vs_merge = mergeStdVectors(vs1, vs2);
  VectorOfVectors lbdas_merge;
  {
    for (size_t i = 0; i < lbdas1.size(); i++) {
      lbdas_merge.push_back(lbdas1[i]);
    }
    lbdas_merge.push_back(thtopt);
    for (size_t i = 1; i < lbdas2.size(); i++) {
      lbdas_merge.push_back(lbdas2[i]);
    }
  }

  BOOST_CHECK_EQUAL(xs1.size(), t0);
  BOOST_CHECK_EQUAL(xs2.size(), t1 + 1);

  VectorXs x_errs(horizon + 1);

  const auto &xs = _sol_full[0];
  for (uint i = 0; i <= horizon; i++) {
    x_errs[i] = infty_norm(xs[i] - xs_merge[i]);
  }
  fmt::print("errors between solves: {}\n", aligator::math::infty_norm(x_errs));

  KktError _err_merged =
      compute_kkt_error(problem, xs_merge, us_merge, vs_merge, lbdas_merge);
  fmt::print("KKT error (merged): {:.3e}\n", _err_merged.max);
  printError(_err_merged, "KKT error (merged)");
}

BOOST_AUTO_TEST_CASE(parallel_solver_class) {
  uint nx = 2;
  uint nu = 2;
  VectorXs x0;
  x0.setRandom(nx);
  uint horizon = 12;

  problem_t problem = generate_problem(x0, horizon, nx, nu);
}
