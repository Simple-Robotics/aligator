#include "aligator/utils/newton-raphson.hpp"

#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(utils)

using namespace aligator;
using Scalar = double;
ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

BOOST_AUTO_TEST_CASE(newton_raphson) {
  const long nx = 4;
  using NR_t = NewtonRaphson<Scalar>;
  auto fun = [](const ConstVectorRef &x, VectorRef out) {
    out = x.array() * x.array() - 1.;
  };
  auto jac_fun = [](const ConstVectorRef &x, MatrixRef out) {
    out.setZero();
    out.diagonal().array() = 2. * x.array();
  };

  VectorXs xinit = 0.1 * VectorXs::Constant(nx, 0.1);
  VectorXs xout(xinit);
  proxsuite::nlp::VectorSpaceTpl<Scalar> space(nx);
  // buffer for evaluating the function
  VectorXs err(nx);
  err.setZero();
  // buffer for direction
  VectorXs dx = err;
  // buffer for jacobian
  MatrixXs jacobian(nx, nx);
  jacobian.setZero();
  // desired precision
  Scalar eps = 1e-6;

  bool conv =
      NR_t::run(space, fun, jac_fun, xinit, xout, err, dx, jacobian, eps, 10);

  // actual answer
  VectorXs xans = VectorXs::Ones(nx);

  fmt::print("Found answer: {}\n", xout.transpose());
  fmt::print("True answer: {}\n", xans.transpose());

  BOOST_TEST_CHECK(conv);
  BOOST_TEST_CHECK(xout.isApprox(xans, eps));
}

BOOST_AUTO_TEST_SUITE_END()
