#include "proxddp/utils/newton-raphson.hpp"

#include <proxnlp/modelling/spaces/vector-space.hpp>

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(utils)

using namespace proxddp;
using Scalar = double;
PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

BOOST_AUTO_TEST_CASE(newton_raphson) {
  const long nx = 4;
  MatrixXs jacobian_(nx, nx);
  auto fun = [](const ConstVectorRef &x) {
    fmt::print("x = {}\n", x.transpose());
    VectorXs out = x.array() * x.array() - 1.;
    fmt::print("out = {}\n", out.transpose());
    return out;
  };
  auto Jfun = [&](const ConstVectorRef &x) {
    jacobian_.setIdentity();
    jacobian_.diagonal().array() *= 2. * x.array();
    return jacobian_;
  };

  VectorXs xinit(nx);
  xinit.setOnes();
  xinit *= 0.1;
  VectorXs xout(xinit);
  proxnlp::VectorSpaceTpl<Scalar> space(nx);
  Scalar eps = 1e-6;
  bool conv =
      NewtonRaphson<Scalar>::run(space, fun, Jfun, xinit, xout, eps, 10);

  VectorXs x_ans(nx);
  x_ans.setOnes();

  fmt::print("Found answer: {}\n", xout.transpose());
  fmt::print("True answer: {}\n", x_ans.transpose());

  BOOST_TEST_CHECK(conv);
  BOOST_TEST_CHECK(xout.isApprox(x_ans, eps));
}

BOOST_AUTO_TEST_SUITE_END()
