/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

#include "./test_util.hpp"

using namespace aligator::gar;

BOOST_AUTO_TEST_SUITE(prox_riccati)

BOOST_AUTO_TEST_CASE(inplace_llt) {
  uint N = 7;
  MatrixXs a(N, N);
  a.setRandom();
  MatrixXs M = a.transpose() * a;

  fmt::print("Matrix M=\n{}\n", M);

  Eigen::LLT<MatrixRef> Mchol(M);
  MatrixXs L(Mchol.matrixL());
  fmt::print("Factor L=\n{}\n", L);
}

BOOST_AUTO_TEST_CASE(short_horz_pb) {
  // dual regularization parameters
  const double mu = 1e-14;
  const double mueq = mu;

  uint nx = 2, nu = 2;
  VectorXs x0 = VectorXs::Ones(nx);
  VectorXs x1 = -VectorXs::Ones(nx);
  auto init_knot = [&]() {
    knot_t knot(nx, nu, 0);
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

  std::vector<knot_t> knots(N + 1, base_knot);
  knots[N] = knot1;
  LQRProblemTpl<double> prob(knots, nx);
  prob.g0 = -x0;
  prob.G0.setIdentity();
  prox_riccati_t solver{prob};
  fmt::print("Horizon: {:d}\n", prob.horizon());
  BOOST_CHECK(solver.backward(mu, mueq));

  auto _traj = lqrInitializeSolution(prob);
  VectorOfVectors xs = std::move(_traj[0]);
  VectorOfVectors us = std::move(_traj[1]);
  VectorOfVectors vs = std::move(_traj[2]);
  VectorOfVectors lbdas = std::move(_traj[3]);
  BOOST_CHECK_EQUAL(xs.size(), prob.horizon() + 1);
  BOOST_CHECK_EQUAL(vs.size(), prob.horizon() + 1);
  BOOST_CHECK_EQUAL(lbdas.size(), prob.horizon() + 1);

  bool ret = solver.forward(xs, us, vs, lbdas);
  BOOST_CHECK(ret);

  // check error
  KktError err = compute_kkt_error(prob, xs, us, vs, lbdas);

  fmt::print("dyn  error: {:.3e}\n", err.dyn);
  fmt::print("cstr error: {:.3e}\n", err.cstr);
  fmt::print("dual error: {:.3e}\n", err.dual);

  BOOST_CHECK_LE(err.max, 1e-9);
}

BOOST_AUTO_TEST_CASE(random_long_problem) {
  Eigen::Vector2d x0 = {0.1, 1.0};
  uint nx = 2;
  uint nu = 3;
  auto prob = generate_problem(x0, 100, nx, nu);
  prox_riccati_t solver{prob};
  double mu = 1e-14;
  solver.backward(mu, mu);

  auto _traj = lqrInitializeSolution(prob);
  VectorOfVectors xs = std::move(_traj[0]);
  VectorOfVectors us = std::move(_traj[1]);
  VectorOfVectors vs = std::move(_traj[2]);
  VectorOfVectors lbdas = std::move(_traj[3]);
  solver.forward(xs, us, vs, lbdas);

  KktError err = compute_kkt_error(prob, xs, us, vs, lbdas);
  fmt::print("dyn  error: {:.3e}\n", err.dyn);
  fmt::print("cstr error: {:.3e}\n", err.cstr);
  fmt::print("dual error: {:.3e}\n", err.dual);

  BOOST_CHECK_LE(err.max, 1e-9);
}

BOOST_AUTO_TEST_CASE(parametric) {
  // TODO: implement a parametric problem and solve it
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()
