#include "aligator/utils/newton-raphson.hpp"
#include "aligator/core/vector-space.hpp"
#include "aligator/fmt-eigen.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace aligator;
using Scalar = double;
ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

TEST_CASE("newton_raphson", "[utils]") {
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
  aligator::VectorSpaceTpl<Scalar> space(nx);
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

  REQUIRE(conv);
  REQUIRE(xout.isApprox(xans, eps));
}
