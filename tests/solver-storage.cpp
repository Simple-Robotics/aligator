#include "proxddp/core/solver-workspace.hpp"

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(solver_workspace)

BOOST_AUTO_TEST_CASE(prox_storage) {
  using value_store_t = proxddp::internal::value_storage<double>;
  using q_store_t = proxddp::internal::q_storage<double>;
  const int NX = 3;
  const int NU = 2;
  auto vstore = value_store_t(NX);
  auto qstore = q_store_t(NX, NU, NX);

  Eigen::Map<Eigen::VectorXd> map(vstore.storage.data(), vstore.storage.size());
  map.setLinSpaced(0., (double)vstore.storage.size() - 1.);
  Eigen::Map<Eigen::VectorXd> map_q(qstore.storage.data(),
                                    qstore.storage.size());
  map_q.setLinSpaced(0., (double)qstore.storage.size() - 1.);

  BOOST_TEST_MESSAGE("Checking value function");
  BOOST_CHECK_EQUAL(vstore.storage.cols(), NX + 1);
  BOOST_CHECK_EQUAL(vstore.Vx().cols(), 1);
  BOOST_CHECK_EQUAL(vstore.Vx().rows(), NX);
  BOOST_CHECK_EQUAL(vstore.Vxx().rows(), NX);
  BOOST_CHECK_EQUAL(vstore.Vxx().cols(), NX);
  fmt::print("{} < store\n", vstore.storage);
  fmt::print("{} < Vx\n", vstore.Vx());
  fmt::print("{} < Vxx\n", vstore.Vxx());

  BOOST_TEST_MESSAGE("Checking Q-function");
  BOOST_CHECK_EQUAL(qstore.storage.cols(), NX * 2 + NU + 1);
  fmt::print("{} < qstore\n", qstore.storage);
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
