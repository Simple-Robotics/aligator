/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
/// @author Wilson Jallet
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

#include "./test_util.hpp"
#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/proximal-riccati.hpp"
#include "aligator/gar/utils.hpp"
#include "aligator/threads.hpp"
#include <Eigen/Cholesky>

using namespace aligator::gar;

constexpr double EPS = 1e-9;
const uint num_threads = 6;

std::array<problem_t, 2> splitProblemInTwo(const problem_t &problem, uint t0,
                                           double mu) {
  assert(problem.isInitialized());
  uint N = (uint)problem.horizon();
  assert(t0 < N);

  problem_t::KnotVector knots1, knots2;
  uint nx_t0 = problem.stages[t0].nx;

  for (uint i = 0; i < t0; i++)
    knots1.push_back(problem.stages[i]);

  for (uint i = t0; i <= N; i++)
    knots2.push_back(problem.stages[i]);

  knot_t kn1_last = knots1.back(); // copy

  problem_t p1(std::move(knots1), problem.nc0());
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

  problem_t p2(std::move(knots2), 0);
  p2.addParameterization(nx_t0);
  {
    knot_t &p2_first = p2.stages[0];
    p2_first.Gx = kn1_last.E.transpose();
  }

  return {std::move(p1), std::move(p2)};
}

/// Test the max-only formulation, where both legs are parameterized
/// by the splitting variable (= costate at t0)
BOOST_AUTO_TEST_CASE(parallel_manual) {
  uint nx = 2;
  uint nu = 2;
  VectorXs x0;
  x0.setRandom(nx);
  const uint horizon = 16;
  const double mu = 1e-14;

  problem_t problem = generate_problem(x0, horizon, nx, nu);

  BOOST_TEST_MESSAGE("Classical solve");
  ProximalRiccatiSolver<double> solver_full_horz(problem);
  solver_full_horz.backward(mu, mu);

  auto solution_full_horz = lqrInitializeSolution(problem);
  auto &[xs, us, vs, lbdas] = solution_full_horz;
  {
    bool ret = solver_full_horz.forward(xs, us, vs, lbdas);
    BOOST_CHECK(ret);
    KktError err_full = computeKktError(problem, xs, us, vs, lbdas);
    fmt::println("KKT error (full horz.): {}", err_full);
    BOOST_CHECK_LE(err_full.max, EPS);
  }

  BOOST_TEST_MESSAGE("Solve in parallel");
  uint t0 = horizon / 2;
  auto prs = splitProblemInTwo(problem, t0, mu);
  auto &pr1 = prs[0];
  auto &pr2 = prs[1];

  BOOST_CHECK_EQUAL(pr1.horizon() + pr2.horizon() + 1, horizon);

  ProximalRiccatiSolver<double> subSolve1(pr1);
  ProximalRiccatiSolver<double> subSolve2(pr2);

  auto sol_leg1 = lqrInitializeSolution(pr1);
  auto sol_leg2 = lqrInitializeSolution(pr2);
  auto &[xs1, us1, vs1, lbdas1] = sol_leg1;
  auto &[xs2, us2, vs2, lbdas2] = sol_leg2;
  MatrixXs thHess = MatrixXs::Zero(nx, nx);
  VectorXs thGrad = VectorXs::Zero(nx);
  VectorXs thtopt = VectorXs::Zero(nx);
  Eigen::LDLT<MatrixXs> hessChol(nx);

  fmt::print("Available threads: {:d}\n",
             aligator::omp::get_available_threads());

#pragma omp parallel sections num_threads(2)
  {
    fmt::print("Current threads: {:d}\n", aligator::omp::get_current_threads());

#pragma omp section
    subSolve1.backward(mu, mu);
#pragma omp section
    subSolve2.backward(mu, mu);
  }
  {
    ALIGATOR_NOMALLOC_END;
    thHess = subSolve1.thHess + subSolve2.thHess;
    thGrad = subSolve1.thGrad + subSolve2.thGrad;
    hessChol.compute(thHess);
    thtopt = hessChol.solve(-thGrad);
    fmt::print("Theta system err = {:.3e}\n",
               infty_norm(thHess * thtopt + thGrad));
  }
  {
    subSolve1.forward(xs1, us1, vs1, lbdas1, thtopt);
    subSolve2.forward(xs2, us2, vs2, lbdas2, thtopt);
  }

  KktError err1 = computeKktError(pr1, xs1, us1, vs1, lbdas1, thtopt);
  KktError err2 = computeKktError(pr2, xs2, us2, vs2, lbdas2, thtopt);

  BOOST_CHECK_LE(err1.max, EPS);
  BOOST_CHECK_LE(err2.max, EPS);

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
  VectorXs u_errs(horizon);
  VectorXs l_errs(horizon + 1);

  for (uint i = 0; i <= horizon; i++) {
    x_errs[i] = infty_norm(xs[i] - xs_merged[i]);
    l_errs[i] = infty_norm(lbdas[i] - lbdas_merged[i]);
    if (i < horizon)
      u_errs[i] = infty_norm(us[i] - us_merged[i]);
  }
  fmt::print("errors between solves: x={:.3e}, u={:.3e}, λ={:.3e}\n",
             infty_norm(x_errs), infty_norm(u_errs), infty_norm(l_errs));

  KktError err_merged =
      computeKktError(problem, xs_merged, us_merged, vs_merged, lbdas_merged);
  fmt::println("KKT error (merged) {}", err_merged);
}

auto sample_normal(Eigen::Index n) {
  return VectorXs::NullaryExpr(n, normal_unary_op{});
}

auto sample_normal(Eigen::Index n, Eigen::Index m) {
  return MatrixXs::NullaryExpr(n, m, normal_unary_op{});
}

/// Randomize some of the parameters of the problem. This simulates something
/// like updating the LQ problem in SQP.
void randomlyModifyProblem(problem_t &prob) {
  auto N = size_t(prob.horizon());
  std::vector<size_t> idx = {0, N / 3, N / 2, N / 2 + 1, N / 2 + 2, N};
  for (auto i : idx) {
    auto &kn = prob.stages.at(i);
    kn.A = sample_normal(kn.nx, kn.nx);
    kn.B.setRandom();
    kn.q = sample_normal(kn.nx);
    kn.R.setIdentity();
    kn.S.setZero();
  }
}

BOOST_AUTO_TEST_CASE(parallel_solver_class) {
  BOOST_TEST_MESSAGE("parallel_solver_class");
  uint nx = 32;
  uint nu = 12;
  VectorXs x0;
  x0.setZero(nx);
  uint horizon = 50;

  const double tol = 1e-10;

  problem_t problem = generate_problem(x0, horizon, nx, nu);
  const problem_t problemRef{problem};
  BOOST_CHECK(problem.get_allocator() == problemRef.get_allocator());
  const double mu = 1e-9;

  auto solutionRef = lqrInitializeSolution(problemRef);
  auto [xs_ref, us_ref, vs_ref, lbdas_ref] = solutionRef;
  auto [xs, us, vs, lbdas] = solutionRef;

  BOOST_TEST_MESSAGE("Run Serial solver (reference solution)");
  ProximalRiccatiSolver<double> refSolver{problemRef};

  {
    refSolver.backward(mu, mu);
    refSolver.forward(xs_ref, us_ref, vs_ref, lbdas_ref);
    KktError err_ref =
        computeKktError(problemRef, xs_ref, us_ref, vs_ref, lbdas_ref, mu, mu);
    fmt::println("{}", err_ref);
    BOOST_CHECK_LE(err_ref.max, tol);
  }

  BOOST_CHECK(problem.isApprox(problemRef));

  BOOST_TEST_MESSAGE("Run Parallel solver");
  ParallelRiccatiSolver<double> parSolver(problem, num_threads);

  {
    parSolver.backward(mu, mu);
    parSolver.forward(xs, us, vs, lbdas);
    KktError err = computeKktError(problem, xs, us, vs, lbdas, mu, mu);
    fmt::println("{}", err);
    BOOST_CHECK_LE(err.max, tol);
  }

  VectorXs xerrs = VectorXs::Zero(horizon + 1);
  VectorXs lerrs = xerrs;
  for (uint i = 0; i <= horizon; i++) {
    xerrs[(long)i] = infty_norm(xs[i] - xs_ref[i]);
    lerrs[(long)i] = infty_norm(lbdas[i] - lbdas_ref[i]);
  }
  for (uint i = 0; i <= horizon; i++) {
    fmt::print("ex[{:d}] = {}\n", i, xerrs[long(i)]);
  }
  for (uint i = 0; i <= horizon; i++) {
    fmt::print("eλ[{:d}] = {}\n", i, lerrs[long(i)]);
  }
  double xerr = infty_norm(xerrs);
  double lerr = infty_norm(lerrs);
  fmt::print("xerrs = {}\n", xerr);
  fmt::print("lerrs = {}\n", lerr);
  BOOST_CHECK_LE(xerr, tol);
  BOOST_CHECK_LE(lerr, tol);

  BOOST_TEST_MESSAGE("Run Parallel solver again [tweak the problem]");
  for (size_t i = 0; i < 10; i++) {
    randomlyModifyProblem(problem);
    parSolver.backward(mu, mu);
    parSolver.forward(xs, us, vs, lbdas);
    KktError e = computeKktError(problem, xs, us, vs, lbdas, mu, mu, false);
    fmt::println("{}", e);
    BOOST_CHECK_LE(e.max, tol);
  }
}
