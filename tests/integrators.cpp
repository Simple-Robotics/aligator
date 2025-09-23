#include "aligator/modelling/dynamics/integrator-euler.hpp"
#include "aligator/modelling/dynamics/integrator-rk2.hpp"
#include "aligator/core/vector-space.hpp"

#include <catch2/catch_test_macros.hpp>

using Manifold = aligator::VectorSpaceTpl<double>;

TEST_CASE("euler", "[integrators]") {
  constexpr int NX = 3;
  Manifold space(NX);
}

TEST_CASE("rk2", "[integrators]") {
  constexpr int NX = 3;
  Manifold space(NX);
}
