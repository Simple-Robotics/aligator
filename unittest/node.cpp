#include "proxddp/node.hpp"

#include "proxnlp/modelling/spaces/vector-space.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(node)

using namespace proxddp;

using Scalar = double;

BOOST_AUTO_TEST_CASE(test_node1)
{
  using Manifold = proxnlp::VectorSpaceTpl<Scalar>;
  using Node = StageModelTpl<Scalar>;

  constexpr int N = 4;

  Manifold space(N);
  shared_ptr<Manifold> space_ptr(&space);
  Node node(space_ptr);

}


BOOST_AUTO_TEST_SUITE_END()
