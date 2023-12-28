/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

#include "./test_util.hpp"
#include "aligator/gar/parallel-solver.hpp"

using namespace aligator::gar;

constexpr double EPS = 1e-9;

std::array<problem_t, 2> splitProblemInTwo(const problem_t &problem, uint t0,
                                           double mu) {
  assert(problem.isInitialized());
  uint N = (uint)problem.horizon();
  assert(t0 < N);

  std::vector<knot_t> knots1, knots2;
  uint nx_t0 = problem.stages[t0].nx;

  for (uint i = 0; i < t0; i++)
    knots1.push_back(problem.stages[i]);

  for (uint i = t0; i <= N; i++)
    knots2.push_back(problem.stages[i]);

  knot_t kn1_last = knots1.back(); // copy

  problem_t p1(knots1, problem.nc0());
  p1.G0 = problem.G0;
  p1.g0 = problem.g0;
  p1.addParameterization(nx_t0);
  {
    knot_t &p1_last = p1.stages.back();
    p1_last.Gx = kn1_last.A.transpose();
    p1_last.Gu = kn1_last.B.transpose();
    p1_last.gamma = kn1_last.f;
    p1_last.Gth.diagonal().setConstant(-mu);
    kn1_last.A.setZero();
    kn1_last.B.setZero();
    kn1_last.f.setZero();
  }

  problem_t p2(knots2, 0);
  p2.addParameterization(nx_t0);
  {
    knot_t &p2_first = p2.stages[0];
    p2_first.Gx = kn1_last.E.transpose();
  }

  return {p1, p2};
}

/// Test the max-only formulation, where both legs are parameterized
/// by the splitting variable (= costate at t0)
BOOST_AUTO_TEST_CASE(parallel_manual) {
  uint nx = 2;
  uint nu = 2;
  VectorXs x0;
  x0.setRandom(nx);
  uint horizon = 16;
  const double mu = 1e-14;

  problem_t problem = generate_problem(x0, horizon, nx, nu);

  BOOST_TEST_MESSAGE("Classical solve");
  prox_riccati_t solver_full_horz(problem);
  solver_full_horz.backward(mu, mu);

  auto solution_full_horz = lqrInitializeSolution(problem);
  {
    auto &xs = solution_full_horz[0];
    auto &us = solution_full_horz[1];
    auto &vs = solution_full_horz[2];
    auto &lbdas = solution_full_horz[3];
    bool ret = solver_full_horz.forward(xs, us, vs, lbdas);
    BOOST_CHECK(ret);
    KktError err_full = compute_kkt_error(problem, xs, us, vs, lbdas);
    printError(err_full, "KKT error (full horz.)");
    BOOST_CHECK_LE(err_full.max, EPS);
  }

  BOOST_TEST_MESSAGE("Solve in parallel");
  uint t0 = horizon / 2;
  auto prs = splitProblemInTwo(problem, t0, mu);
  auto &pr1 = prs[0];
  auto &pr2 = prs[1];

  BOOST_CHECK_EQUAL(pr1.horizon() + pr2.horizon() + 1, horizon);

  prox_riccati_t subSolve1(pr1);
  prox_riccati_t subSolve2(pr2);

  auto sol_leg1 = lqrInitializeSolution(pr1);
  auto sol_leg2 = lqrInitializeSolution(pr2);
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
    subSolve1.forward(sol_leg1[0], sol_leg1[1], sol_leg1[2], sol_leg1[3],
                      ConstVectorRef(thtopt));
    // subSolve2.forward(_sol2[0], _sol2[1], _sol2[2], _sol2[3]);
    subSolve2.forward(sol_leg2[0], sol_leg2[1], sol_leg2[2], sol_leg2[3],
                      ConstVectorRef(thtopt));
  }
  auto &xs1 = sol_leg1[0];
  auto &us1 = sol_leg1[1];
  auto &vs1 = sol_leg1[2];
  auto &lbdas1 = sol_leg1[3];
  auto &xs2 = sol_leg2[0];
  auto &us2 = sol_leg2[1];
  auto &vs2 = sol_leg2[2];
  auto &lbdas2 = sol_leg2[3];

  KktError err1 = compute_kkt_error(pr1, xs1, us1, vs1, lbdas1, thtopt);
  KktError err2 = compute_kkt_error(pr2, xs2, us2, vs2, lbdas2, thtopt);

  BOOST_CHECK_LE(err1.max, EPS);
  BOOST_CHECK_LE(err2.max, EPS);

  fmt::print("err1 = {:.3e}\n", err1.max);
  fmt::print("err2 = {:.3e}\n", err2.max);

  uint t1 = horizon - t0;

  VectorOfVectors xs_merged = mergeStdVectors(xs1, xs2);
  VectorOfVectors us_merged = mergeStdVectors(us1, us2);
  VectorOfVectors vs_merged = mergeStdVectors(vs1, vs2);
  VectorOfVectors lbdas_merged;
  {
    for (size_t i = 0; i < lbdas1.size(); i++) {
      lbdas_merged.push_back(lbdas1[i]);
    }
    lbdas_merged.push_back(thtopt);
    for (size_t i = 1; i < lbdas2.size(); i++) {
      lbdas_merged.push_back(lbdas2[i]);
    }
  }

  BOOST_CHECK_EQUAL(xs1.size(), t0);
  BOOST_CHECK_EQUAL(xs2.size(), t1 + 1);

  VectorXs x_errs(horizon + 1);

  const auto &xs = solution_full_horz[0];
  for (uint i = 0; i <= horizon; i++) {
    x_errs[i] = infty_norm(xs[i] - xs_merged[i]);
  }
  fmt::print("errors between solves: {}\n", infty_norm(x_errs));

  KktError err_merged =
      compute_kkt_error(problem, xs_merged, us_merged, vs_merged, lbdas_merged);
  printError(err_merged, "KKT error (merged)");
}

BOOST_AUTO_TEST_CASE(parallel_solver_class) {
  BOOST_TEST_MESSAGE("parallel_solver_class");
  uint nx = 2;
  uint nu = 2;
  VectorXs x0;
  x0.setRandom(nx);
  uint horizon = 12;

  problem_t problem = generate_problem(x0, horizon, nx, nu);
}
