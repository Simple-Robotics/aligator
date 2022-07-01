#include "proxddp/core/solver-workspace.hpp"

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(solver_workspace)

BOOST_AUTO_TEST_CASE(storage) {
  using value_store_t = proxddp::internal::value_storage<double>;
  using q_store_t = proxddp::internal::q_function_storage<double>;
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
  BOOST_CHECK_EQUAL(vstore.Vx_.cols(), 1);
  BOOST_CHECK_EQUAL(vstore.Vx_.rows(), NX);
  BOOST_CHECK_EQUAL(vstore.Vxx_.rows(), NX);
  BOOST_CHECK_EQUAL(vstore.Vxx_.cols(), NX);
  fmt::print("{} < store\n", vstore.storage);
  fmt::print("{} < Vx\n", vstore.Vx_);
  fmt::print("{} < Vxx\n", vstore.Vxx_);

  BOOST_TEST_MESSAGE("Checking Q-function");
  BOOST_CHECK_EQUAL(qstore.storage.cols(), NX * 2 + NU + 1);
  fmt::print("{} < qstore\n", qstore.storage);
  fmt::print("{} < Qx\n", qstore.Qx_);
  fmt::print("{} < Qu\n", qstore.Qu_);
  fmt::print("{} < Qy\n", qstore.Qy_);
  fmt::print("{} < Qxx\n", qstore.Qxx_);
  fmt::print("{} < Qxu\n", qstore.Qxu_);
  fmt::print("{} < Qxy\n", qstore.Qxy_);
  fmt::print("{} < Quu\n", qstore.Quu_);
  fmt::print("{} < Qyy\n", qstore.Qyy_);
}

BOOST_AUTO_TEST_SUITE_END()
