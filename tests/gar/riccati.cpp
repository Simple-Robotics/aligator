/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

#include "./util.hpp"

using namespace aligator::gar;

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

BOOST_AUTO_TEST_CASE(proxriccati) {
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

  auto _sol = lqrInitializeSolution(prob);
  std::vector<VectorXs> &xs = _sol[0];
  std::vector<VectorXs> &us = _sol[1];
  std::vector<VectorXs> &vs = _sol[2];
  std::vector<VectorXs> &lbdas = _sol[3];

  bool ret = solver.forward(xs, us, vs, lbdas);
  BOOST_CHECK(ret);

  // check error
  KktError err = compute_kkt_error(prob, xs, us, vs, lbdas);

  fmt::print("dyn  error: {:.3e}\n", err.dyn);
  fmt::print("cstr error: {:.3e}\n", err.cstr);
  fmt::print("dual error: {:.3e}\n", err.dual);

  BOOST_CHECK_LE(max_kkt_error(err), 1e-8);
}

BOOST_AUTO_TEST_CASE(parametric) {
  // TODO: implement a parametric problem and solve it
  BOOST_CHECK(true);
}
