#include "aligator/solvers/value-function.hpp"

#include <catch2/catch_test_macros.hpp>

#include <aligator/fmt-eigen.hpp>

TEST_CASE("prox_storage", "[solver_workspace]") {
  using VParams = aligator::ValueFunctionTpl<double>;
  using QParams = aligator::QFunctionTpl<double>;
  const int NX = 3;
  const int NU = 2;
  VParams vstore(NX);
  QParams qstore(NX, NU, NX);

  REQUIRE(vstore.Vx_.cols() == 1);
  REQUIRE(vstore.Vx_.rows() == NX);
  REQUIRE(vstore.Vxx_.rows() == NX);
  REQUIRE(vstore.Vxx_.cols() == NX);
  fmt::print("{} < Vx\n", vstore.Vx_);
  fmt::print("{} < Vxx\n", vstore.Vxx_);

  fmt::print("{} < Qx\n", qstore.Qx);
  fmt::print("{} < Qu\n", qstore.Qu);
  fmt::print("{} < Qxx\n", qstore.Qxx);
  fmt::print("{} < Qxu\n", qstore.Qxu);
  fmt::print("{} < Qxy\n", qstore.Qxy);
  fmt::print("{} < Quu\n", qstore.Quu);
  fmt::print("{} < Qyy\n", qstore.Qyy);
}

TEST_CASE("fddp_storage", "[solver_workspace]") {}
