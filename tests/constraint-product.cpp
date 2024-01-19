#include "aligator/core/constraint-set-product.hpp"
#include <proxsuite-nlp/modelling/constraints.hpp>

#include <boost/test/unit_test.hpp>

using namespace aligator;
using proxsuite::nlp::EqualityConstraint;
using proxsuite::nlp::NegativeOrthant;
using CstrProdOp = ConstraintSetProductTpl<double>;
using Eigen::MatrixXd;
using Eigen::VectorXd;

BOOST_AUTO_TEST_CASE(constraint_product_op) {
  EqualityConstraint<double> eq_op;
  NegativeOrthant<double> neg_op;
  long n1 = 2, n2 = 3;
  CstrProdOp op;
  op.components = {&eq_op, &neg_op};
  op.nrs = {n1, n2};

  VectorXd z(n1 + n2);
  z.setRandom();
  VectorXd zCopy = z;

  op.normalConeProjection(zCopy, z);
  fmt::print("zproj = {}", z.transpose());

  neg_op.normalConeProjection(zCopy.tail(n2), zCopy.tail(n2));

  BOOST_CHECK(z.isApprox(zCopy));
}
