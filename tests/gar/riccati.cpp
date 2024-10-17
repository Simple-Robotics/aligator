/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

// used for timings, numbers are merely indicative, this *not* a benchmark
#include <chrono>

#include "aligator/gar/utils.hpp"
#include "aligator/gar/proximal-riccati.hpp"
#include "aligator/gar/dense-riccati.hpp"

#include "test_util.hpp"
#include <proxsuite-nlp/fmt-eigen.hpp>

using namespace aligator::gar;

BOOST_AUTO_TEST_CASE(short_horz_pb) {
  // dual regularization parameters
  const double mu = 1e-14;
  const double mueq = mu;

  uint nx = 2, nu = 2;
  VectorXs x0 = VectorXs::Ones(nx);
  VectorXs x1 = -VectorXs::Ones(nx);
  auto init_knot = [&](uint nc = 0) {
    knot_t knot(nx, nu, nc);
    knot.A << 0.1, 0., -0.1, 0.01;
    knot.B.setRandom();
    knot.E.setIdentity();
    knot.E *= -1;
    knot.f.setRandom();
    knot.Q.setIdentity();
    knot.Q *= 0.01;
    knot.R.setIdentity();
    knot.R *= 0.1;
    return knot;
  };
  auto base_knot = init_knot();
  auto knot1 = base_knot;
  // knot1.d = -x1;
  {
    knot1.Q.setIdentity();
    knot1.q = -x1;
  }

  uint N = 8;
  problem_t::KnotVector knots(N + 1, base_knot);
  knots[4] = init_knot(nu);
  knots[4].D.setIdentity();
  knots[4].d.setConstant(0.1);
  knots[N] = knot1;
  problem_t prob(knots, nx);
  prob.g0 = -x0;
  prob.G0.setIdentity();
  ProximalRiccatiSolver<double> solver{prob};
  fmt::print("Horizon: {:d}\n", prob.horizon());

  auto bwbeg = std::chrono::system_clock::now();
  BOOST_CHECK(solver.backward(mu, mueq));
  auto bwend = std::chrono::system_clock::now();
  auto t_bwd =
      std::chrono::duration_cast<std::chrono::microseconds>(bwend - bwbeg);
  fmt::print("Elapsed time (bwd): {:d}\n", t_bwd.count());

  auto [xs, us, vs, lbdas] = lqrInitializeSolution(prob);
  BOOST_CHECK_EQUAL(xs.size(), prob.horizon() + 1);
  BOOST_CHECK_EQUAL(vs.size(), prob.horizon() + 1);
  BOOST_CHECK_EQUAL(lbdas.size(), prob.horizon() + 1);

  auto fwbeg = std::chrono::system_clock::now();
  bool ret = solver.forward(xs, us, vs, lbdas);
  auto fwend = std::chrono::system_clock::now();
  auto t_fwd =
      std::chrono::duration_cast<std::chrono::microseconds>(fwend - fwbeg);
  fmt::print("Elapsed time (fwd): {:d}\n", t_fwd.count());
  BOOST_CHECK(ret);

  // check error
  KktError err = computeKktError(prob, xs, us, vs, lbdas);

  fmt::println("{}", err);

  BOOST_CHECK_LE(err.max, 1e-9);

  for (size_t i = 0; i < N; i++) {
    fmt::print("us[{:>2d}] = {}\n", i, us[i].transpose());
  }
}

BOOST_AUTO_TEST_CASE(one_knot_prob) {
  BOOST_TEST_MESSAGE("Single-knot problem");
  uint nx = 2;
  uint nu = 2;
  Eigen::VectorXd x0;
  x0.setZero(nx);
  auto problem = generate_problem(x0, 0, nx, nu);
  ProximalRiccatiSolver<double> solver(problem);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  BOOST_CHECK_EQUAL(xs.size(), 1);
  BOOST_CHECK_EQUAL(us.size(), 0);
  BOOST_CHECK_EQUAL(lbdas.size(), 1);
  double mu = 1e-13;
  solver.backward(mu, mu);
  solver.forward(xs, us, vs, lbdas);

  KktError err = computeKktError(problem, xs, us, vs, lbdas);
  BOOST_CHECK_LE(err.max, 1e-10);
}

BOOST_AUTO_TEST_CASE(random_long_problem) {
  BOOST_TEST_MESSAGE("Random long problem");
  uint nx = 36;
  uint nu = 12;
  Eigen::VectorXd x0;
  x0.setZero(nx);
  uint horz = 100;
  auto prob = generate_problem(x0, horz, nx, nu);
  ProximalRiccatiSolver<double> solver{prob};
  const double mu = 1e-14;
  auto bwbeg = std::chrono::system_clock::now();
  solver.backward(mu, mu);
  auto bwend = std::chrono::system_clock::now();
  auto t_bwd =
      std::chrono::duration_cast<std::chrono::microseconds>(bwend - bwbeg);
  fmt::print("Elapsed time (bwd): {:d}\n", t_bwd.count());

  auto [xs, us, vs, lbdas] = lqrInitializeSolution(prob);
  auto fwbeg = std::chrono::system_clock::now();
  solver.forward(xs, us, vs, lbdas);
  auto fwend = std::chrono::system_clock::now();
  auto t_fwd =
      std::chrono::duration_cast<std::chrono::microseconds>(fwend - fwbeg);
  fmt::print("Elapsed time (fwd): {:d}\n", t_fwd.count());

  KktError err = computeKktError(prob, xs, us, vs, lbdas);
  fmt::println("{}", err);

  BOOST_CHECK_LE(err.max, 1e-9);

  BOOST_TEST_MESSAGE("Dense stagewise solver:");
  {
    RiccatiSolverDense<double> denseSolver(prob);
    auto bwbeg = std::chrono::system_clock::now();
    denseSolver.backward(mu, mu);
    auto bwend = std::chrono::system_clock::now();
    auto t_bwd =
        std::chrono::duration_cast<std::chrono::microseconds>(bwend - bwbeg);
    fmt::print("Elapsed time (bwd, dense): {:d}\n", t_bwd.count());
    auto [xsd, usd, vsd, lbdasd] = lqrInitializeSolution(prob);
    denseSolver.forward(xsd, usd, vsd, lbdasd);
    KktError err = computeKktError(prob, xsd, usd, vsd, lbdasd);
    fmt::println("{}", err);
    BOOST_CHECK_LE(err.max, 1e-9);
  }
}

BOOST_AUTO_TEST_CASE(parametric) {
  BOOST_TEST_MESSAGE("parametric");
  Eigen::Vector3d x0 = Eigen::Vector3d::NullaryExpr(normal_unary_op{});
  uint nx = uint(x0.rows());
  uint nu = 2;
  uint horz = 100;
  uint nth = 1;
  auto problem = generate_problem(x0, horz, nx, nu, nth);
  const double mu = 1e-12;
  auto testfn = [&](auto &&solver) {
    auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
    solver.backward(mu, mu);

    VectorXs theta(nth);
    theta.setRandom();
    solver.forward(xs, us, vs, lbdas, theta);
    auto e = xs[0] - x0;
    fmt::print("x0 = {}\n", x0.transpose());
    fmt::print("e = {}\n", e.transpose());

    KktError err = computeKktError(problem, xs, us, vs, lbdas, theta);
    fmt::println("{}", err);
    BOOST_CHECK_LE(err.max, 1e-10);
  };

  BOOST_TEST_MESSAGE("ProximalRiccatiSolver");
  ProximalRiccatiSolver<double> solver(problem);
  testfn(solver);

  using aligator::math::check_value;
  BOOST_TEST_CHECK(!check_value(solver.kkt0.chol.matrixLDLT()));
  BOOST_TEST_CHECK(!check_value(solver.kkt0.ff.matrix()));
  BOOST_TEST_CHECK(!check_value(solver.kkt0.fth.matrix()));
  BOOST_TEST_CHECK(!check_value(solver.thGrad));
  BOOST_TEST_CHECK(!check_value(solver.thHess));
  BOOST_TEST_CHECK(!check_value(solver.datas[0].vm.vt));
  BOOST_TEST_CHECK(!check_value(solver.datas[0].vm.Vxt));
  BOOST_TEST_CHECK(!check_value(solver.datas[0].vm.Vtt));
  BOOST_TEST_CHECK(!check_value(solver.datas[horz].vm.vt));
  BOOST_TEST_CHECK(!check_value(solver.datas[horz].vm.Vxt));
  BOOST_TEST_CHECK(!check_value(solver.datas[horz].vm.Vtt));

  BOOST_TEST_MESSAGE("RiccatiSolverDense");
  RiccatiSolverDense<double> denseSolver(problem);
  testfn(denseSolver);
}
