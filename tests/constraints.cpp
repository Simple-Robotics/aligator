#include "aligator/core/constraint-set.hpp"
#include "aligator/core/vector-space.hpp"
#include "aligator/modelling/constraints.hpp"
#include "aligator/fmt-eigen.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace aligator;
using namespace aligator::context;

const int N = 20;
VectorSpace space(N);

TEST_CASE("test_equality", "[constraint]") {
  VectorXs x0 = space.neutral();
  VectorXs x1 = space.rand();
  VectorXs zout(N);
  zout.setZero();

  EqualityConstraintTpl<double> eq_set;
  double mu = 0.1;

  eq_set.setProxParameter(mu);
  double m = eq_set.computeMoreauEnvelope(x1, zout);
  REQUIRE(zout.isApprox(x1));
  REQUIRE(m == (0.5 / mu * zout.squaredNorm()));
}

TEST_CASE("constraint_product_op", "[constraint]") {
  EqualityConstraintTpl<double> eq_op;
  NegativeOrthantTpl<double> neg_op;
  long n1 = 2, n2 = 3;
  ConstraintSetProductTpl<double> op({eq_op, neg_op}, {n1, n2});

  VectorXs z(n1 + n2);
  z.setRandom();
  z[4] = -0.14;
  VectorXs zCopy = z;

  fmt::print("z = {}\n", z.transpose());
  op.normalConeProjection(zCopy, z);
  fmt::print("zproj = {}\n", z.transpose());

  neg_op.normalConeProjection(zCopy.tail(n2), zCopy.tail(n2));

  REQUIRE(z.isApprox(zCopy));
}
