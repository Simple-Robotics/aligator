#include "proxddp/core/linesearch.hpp"
#include "proxddp/fddp/linesearch.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(linesearch)

auto fun1 = [](const double a) { return a * a - 1.; }

auto dfun1 = [](const double a) { return 2.0 * a; }

BOOST_AUTO_TEST_CASE(goldstein) {}

BOOST_AUTO_TEST_SUITE_END()
