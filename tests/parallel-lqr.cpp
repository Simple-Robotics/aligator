#include <boost/test/unit_test.hpp>

#include "proxddp/parlqr/parlqr.hpp"

using namespace proxddp;

using T = double;

BOOST_AUTO_TEST_CASE(lqrtree) {
  uint nx = 2;
  uint nu = 2;
  uint nc = 0;
  LQRKnot<T> node(nx, nu, nc);
  node.C.setZero();
  node.A.setIdentity();
  node.B.setIdentity();
  if (nc > 0)
    node.C(0, 0) = 1.;
  node.S.setConstant(0.1);

  size_t N_NODES = 16;

  std::vector<LQRKnot<T>> knots;
  knots.assign(N_NODES, node);

  LQRTree<T> tree{knots};

  fmt::print("tree total depth {:d}\n", tree.maxDepth());
  BOOST_CHECK_EQUAL(tree.maxDepth(), (size_t)std::log2(N_NODES));

  auto print_index = [&](size_t depth, size_t i) {
    fmt::print("Index @ (depth={:d}, i={:d}) = {:d}\n", depth, i,
               tree.getIndex(depth, i));
  };
  auto print_children = [&](size_t index) {
    fmt::print("Children of index {} = {}\n", index, tree.getChildren(index));
  };

  print_index(0, 0);
  print_index(1, 0);
  print_index(1, 1);
  print_index(3, 1);

  fmt::print("Leaf index {} = {}\n", 0, tree.getLeafIndex(0));
  fmt::print("Leaf index {} = {}\n", 1, tree.getLeafIndex(1));
  fmt::print("Leaf index {} = {}\n", 4, tree.getLeafIndex(4));

  print_children(0);
  print_children(1);
  print_children(6);

  fmt::print("Depth of index {} = {}\n", 4, tree.getIndexDepth(4));
  fmt::print("Depth of index {} = {}\n", 3, tree.getIndexDepth(3));
  fmt::print("Depth of index {} = {}\n", 9, tree.getIndexDepth(10));

  auto print_parent = [&](size_t index) {
    fmt::print("Parent of index {} = {}\n", index, tree.getIndexParent(index));
  };

  print_parent(0);
  print_parent(4);
  print_parent(14);
  print_parent(15);
}
