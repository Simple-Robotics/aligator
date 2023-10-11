/// Test routines for the parallel LQR solver.
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <boost/test/unit_test.hpp>

#include "aligator/gar/parlqr.hpp"

using namespace aligator;

using T = double;
ALIGATOR_DYNAMIC_TYPEDEFS(T);
using problem_t = gar::LQRProblem<T>;
using solver_t = LQRTreeSolver<T>;
using knot_t = gar::LQRKnot<T>;

BOOST_AUTO_TEST_CASE(factor) {
  uint nx = 2, nu = 2, nc = 1;

  LQRFactor<T> fac{nx, nu, nc, 2};

  fac.X.setConstant(0.1);
  fac.X(1, 1) = -1.41;
  fac.U.setRandom();
  fac.Lambda.setIdentity();
  fac.Nu.setConstant(-3.14);

  fmt::print("{}\n", fac.X);
  fmt::print("{}\n", fac.U);
  fmt::print("{}\n", fac.Lambda);
  fmt::print("{}\n", fac.Nu);

  auto row_mat_concat = fac.data;
  fmt::print("===\n");
  fmt::print("Concat matrix:\n{}\n", row_mat_concat);
}

BOOST_AUTO_TEST_CASE(lqrtree) {
  uint nx = 2;
  uint nu = 2;
  uint nc = 0;
  knot_t node(nx, nu, nc);
  node.C.setZero();
  node.A.setIdentity();
  node.B.setIdentity();
  if (nc > 0)
    node.C(0, 0) = 1.;
  node.S.setConstant(0.1);

  size_t N_NODES = 16;

  std::vector<knot_t> knots;
  knots.assign(N_NODES, node);

  LQRTree tree(N_NODES);

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

void printSolverRhsOrSolution(solver_t const &solver) {
  for (size_t i = 0; i < solver.horz; ++i) {
    fmt::print("{}=\n{}\n", i, solver.rhsAndSol[i].data);
  }
}

void printSolverLhs(solver_t const &solver) {
  for (size_t i = 0; i < solver.horz; ++i) {
    fmt::print("{}=\n{}\n", i, solver.ldlts[i].mat);
  }
}

BOOST_AUTO_TEST_CASE(lqrsolve) {

  uint nx = 2;
  uint nu = 1;
  uint nc = 0;
  knot_t node(nx, nu, nc);
  node.Q.setIdentity();
  node.q.setConstant(10.);
  node.R.setConstant(-1.);
  node.r.setConstant(-2.);

  auto node2 = node;
  node2.Q(0, 0) = 10.;
  node2.r << 0.1;

  std::vector<knot_t> knots = {node, node2};

  problem_t prob{knots};
  solver_t solver(prob);

  fmt::print("lhs:\n");
  printSolverLhs(solver);

  fmt::print("before solve:\n");
  printSolverRhsOrSolution(solver);

  solver.solve();

  fmt::print("sols:\n");
  printSolverRhsOrSolution(solver);

  VectorXs Xsol(nx), Usol(nu);
  Xsol << 10., 10.;
  Usol << 2.;
  BOOST_CHECK_EQUAL(solver.rhsAndSol[0].X, Xsol);
  BOOST_CHECK_EQUAL(solver.rhsAndSol[0].U, Usol);

  Xsol(0) = 1.;
  Usol << -0.1;
  BOOST_CHECK_EQUAL(solver.rhsAndSol[1].X, Xsol);
  BOOST_CHECK_EQUAL(solver.rhsAndSol[1].U, Usol);
}
