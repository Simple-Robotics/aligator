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
  using NR_t = NewtonRaphson<Scalar>;
  auto fun = [](const ConstVectorRef &x, auto &out) {
    fmt::print("x = {}\n", x.transpose());
    out = x.array() * x.array() - 1.;
    fmt::print("out = {}\n", out.transpose());
  };
  auto jac_fun = [&](const ConstVectorRef &x, auto &out) {
    out.setIdentity();
    out.diagonal().array() *= 2. * x.array();
  };

  VectorXs xinit(nx);
  xinit.setOnes();
  xinit *= 0.1;
  VectorXs xout(xinit);
  proxnlp::VectorSpaceTpl<Scalar> space(nx);
  // buffer for evaluating the function
  VectorXs err(nx);
  err.setZero();
  // buffer for direction
  VectorXs dx = err;
  // buffer for jacobian
  MatrixXs jacobian_(nx, nx);
  Scalar eps = 1e-6;
  NR_t::DataView data{err, dx, jacobian_};
  bool conv = NewtonRaphson<Scalar>::run(space, fun, jac_fun, xinit, xout, data,
                                         eps, 10);

  VectorXs x_ans(nx);
  x_ans.setOnes();

  fmt::print("Found answer: {}\n", xout.transpose());
  fmt::print("True answer: {}\n", x_ans.transpose());

  BOOST_TEST_CHECK(conv);
  BOOST_TEST_CHECK(xout.isApprox(x_ans, eps));
}

BOOST_AUTO_TEST_SUITE_END()
