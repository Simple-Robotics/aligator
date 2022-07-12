#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/control.hpp>
#include <crocoddyl/core/states/euclidean.hpp>

#include "proxddp/compat/crocoddyl/problem.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(croc_problem)

namespace pcroc = proxddp::compat::croc;

BOOST_AUTO_TEST_CASE(croc_lqr) {
  std::size_t nx = 4;
  crocoddyl::StateVector state(nx);
}

BOOST_AUTO_TEST_SUITE_END()
