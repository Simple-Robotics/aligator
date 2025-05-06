#include "aligator/core/constraint-set.hpp"
#include "aligator/core/vector-space.hpp"
#include "aligator/modelling/constraints.hpp"
#include "aligator/fmt-eigen.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(constraint)

using namespace aligator;
using namespace aligator::context;

const int N = 20;
VectorSpace space(N);

BOOST_AUTO_TEST_CASE(test_equality) {
  VectorXs x0 = space.neutral();
  VectorXs x1 = space.rand();
  VectorXs zout(N);
  zout.setZero();

  EqualityConstraintTpl<double> eq_set;
  double mu = 0.1;

  eq_set.setProxParameter(mu);
  double m = eq_set.computeMoreauEnvelope(x1, zout);
  BOOST_TEST_CHECK(zout.isApprox(x1));
  BOOST_TEST_CHECK(m == (0.5 / mu * zout.squaredNorm()));
}

BOOST_AUTO_TEST_CASE(constraint_product_op) {
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

  BOOST_CHECK(z.isApprox(zCopy));
}

BOOST_AUTO_TEST_SUITE_END()
