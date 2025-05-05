#include "aligator/modelling/dynamics/integrator-euler.hpp"
#include "aligator/modelling/dynamics/integrator-rk2.hpp"

#include "aligator/core/vector-space.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(integrators)
using Manifold = aligator::VectorSpaceTpl<double>;

BOOST_AUTO_TEST_CASE(euler) {
  constexpr int NX = 3;
  Manifold space(NX);
}

BOOST_AUTO_TEST_CASE(rk2) {
  constexpr int NX = 3;
  Manifold space(NX);
}

BOOST_AUTO_TEST_SUITE_END()
