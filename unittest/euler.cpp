#include "proxddp/modelling/dynamics/euler.hpp"

#include <proxnlp/modelling/spaces/vector-space.hpp>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(integrators)

BOOST_AUTO_TEST_CASE(euler)
{
  using Manifold = proxnlp::VectorSpaceTpl<double>;
  using IntegratorType = proxddp::dynamics::IntegratorEuler<double>;
  constexpr int NX = 3;
  Manifold space(NX);

}


BOOST_AUTO_TEST_SUITE_END()
