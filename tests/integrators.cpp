#include "aligator/modelling/dynamics/integrator-euler.hpp"
#include "aligator/modelling/dynamics/integrator-rk2.hpp"

#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(integrators)

BOOST_AUTO_TEST_CASE(euler) {
  using Manifold = proxsuite::nlp::VectorSpaceTpl<double>;
  constexpr int NX = 3;
  Manifold space(NX);
}

BOOST_AUTO_TEST_CASE(rk2) {
  using Manifold = proxsuite::nlp::VectorSpaceTpl<double>;
  constexpr int NX = 3;
  Manifold space(NX);
}

BOOST_AUTO_TEST_SUITE_END()
