#include "proxddp/core/workspace.hpp"

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(solver_workspace)

BOOST_AUTO_TEST_CASE(prox_storage) {
  using VParams = proxddp::value_function<double>;
  using QParams = proxddp::q_function<double>;
  const int NX = 3;
  const int NU = 2;
  VParams vstore(NX);
  QParams qstore(NX, NU, NX);

  BOOST_TEST_MESSAGE("Checking value function");
  BOOST_CHECK_EQUAL(vstore.Vx_.cols(), 1);
  BOOST_CHECK_EQUAL(vstore.Vx_.rows(), NX);
  BOOST_CHECK_EQUAL(vstore.Vxx_.rows(), NX);
  BOOST_CHECK_EQUAL(vstore.Vxx_.cols(), NX);
  fmt::print("{} < Vx\n", vstore.Vx_);
  fmt::print("{} < Vxx\n", vstore.Vxx_);

  BOOST_TEST_MESSAGE("Checking Q-function");
  fmt::print("{} < Qx\n", qstore.Qx);
  fmt::print("{} < Qu\n", qstore.Qu);
  fmt::print("{} < Qy\n", qstore.Qy);
  fmt::print("{} < Qxx\n", qstore.Qxx);
  fmt::print("{} < Qxu\n", qstore.Qxu);
  fmt::print("{} < Qxy\n", qstore.Qxy);
  fmt::print("{} < Quu\n", qstore.Quu);
  fmt::print("{} < Qyy\n", qstore.Qyy);
}

BOOST_AUTO_TEST_CASE(fddp_storage) {}

BOOST_AUTO_TEST_SUITE_END()
