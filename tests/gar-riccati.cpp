#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

#include "aligator/gar/riccati.hpp"

using namespace aligator;
using namespace gar;

using T = double;
using prox_riccati_t = ProximalRiccatiSolver<T>;
using knot_t = LQRKnot<T>;
ALIGATOR_DYNAMIC_TYPEDEFS(T);

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
  const T mu = 1e-7;
  const T mueq = 1e-6;
  uint nx = 2, nu = 2;
  VectorXs x0 = VectorXs::Ones(nx);
  VectorXs x1 = -VectorXs::Ones(nx);
  knot_t knot0{nx, nu, nx};
  knot_t base_knot{nx, nu, 0};
  auto init_knot = [](knot_t &knot) {
    knot.A << 0.1, 0., -0.1, 0.;
    knot.B.setIdentity();
    knot.E.setIdentity();
    knot.E *= -1;
    knot.Q.setIdentity();
    knot.Q *= 0.01;
    knot.R.setIdentity();
    knot.R *= 0.1;
  };
  init_knot(base_knot);
  init_knot(knot0);
  knot0.C.setIdentity();
  knot0.d = -x0;
  auto knot1 = knot0;
  // knot1.d = -x1;
  {
    knot1.Q.setIdentity();
    knot1.q = -x1;
  }

  uint N = 8;

  std::vector<knot_t> knots(N + 1, base_knot);
  knots[0] = knot0;
  knots[N] = knot1;
  LQRProblem<T> prob(knots, 0);
  prox_riccati_t solver{prob};
  fmt::print("Horizon: {:d}\n", prob.horizon());
  BOOST_CHECK(solver.backward(mu, mueq));

  std::vector<VectorXs> xs{N + 1, VectorXs(nx)};
  std::vector<VectorXs> us{N, VectorXs(nu)};
  std::vector<VectorXs> vs{N + 1, VectorXs::Zero(0)};
  vs[0].resize(knots[0].nc);
  vs[N].resize(knots[N].nc);

  std::vector<VectorXs> lbdas{N, VectorXs::Zero(nx)};

  bool ret = solver.forward(xs, us, vs, lbdas);
  BOOST_CHECK(ret);

  for (uint t = 0; t <= N; t++) {
    fmt::print("xs[{:d}] = {}\n", t, xs[t].transpose());
  }
  for (uint t = 0; t < N; t++) {
    fmt::print("us[{:d}] = {}\n", t, us[t].transpose());
  }
  for (uint t = 0; t <= N; t++) {
    fmt::print("ν[{:d}] = {}\n", t, vs[t].transpose());
  }
  for (uint t = 0; t < N; t++) {
    fmt::print("λ[{:d}] = {}\n", t, lbdas[t].transpose());
  }
}
