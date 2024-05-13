#include "aligator/gar/cholmod-solver.hpp"
#include "aligator/gar/utils.hpp"

#include <boost/test/unit_test.hpp>
#include "test_util.hpp"

using namespace aligator;

BOOST_AUTO_TEST_CASE(helper_assignment_dense) {
  using SparseType = Eigen::SparseMatrix<double>;
  using Eigen::Index;
  constexpr Index n = 20;
  SparseType mat(n, n);
  mat.setZero();
  MatrixXs densemat(n, n);
  densemat.setZero();
  MatrixXs submat(10, 9);
  submat.setRandom();
  densemat.bottomRightCorner(10, 9) = submat;
  gar::helpers::sparseAssignDenseBlock<false>(10, 11, submat, mat);
  BOOST_CHECK(densemat.isApprox(mat.toDense()));
  fmt::println("submat 1:\n{}", submat);

  submat.setRandom(5, 5);
  densemat.block(3, 4, 5, 5) = submat;
  gar::helpers::sparseAssignDenseBlock<false>(3, 4, submat, mat);
  BOOST_CHECK(densemat.isApprox(mat.toDense()));
  fmt::println("submat 2:\n{}", submat);

  fmt::println("dense mat:\n{}", densemat);
  fmt::println("sparse mat:\n{}", mat.toDense());

  // reinitialize matrix

  densemat.setZero();
  mat.setZero();
  densemat.block(3, 4, 5, 5) = submat;
  gar::helpers::sparseAssignDenseBlock<true>(3, 4, submat, mat);
  BOOST_CHECK(densemat.isApprox(mat.toDense()));
}

problem_t short_problem(VectorXs x0, uint horz, uint nx, uint nu, uint nc) {

  problem_t::KnotVector knots{horz + 1};
  const uint nc0 = (uint)x0.size();

  for (uint t = 0; t <= horz; t++) {
    problem_t::KnotType knot{nx, nu, nc};
    knot.Q.setConstant(+0.11);
    knot.R.setConstant(+0.12);
    knot.S.setConstant(-0.2);

    knot.C.setConstant(-2.);
    knot.D.setConstant(-3.);

    knot.f.setConstant(-0.3);
    knot.A.setConstant(0.2);
    knot.B.setConstant(0.3);
    knot.E.setIdentity();
    knot.E *= -1;
    knots[t] = knot;
  }
  auto out = problem_t{knots, nc0};
  out.G0.setConstant(-3.14);
  out.g0.setConstant(42);
  return out;
}

BOOST_AUTO_TEST_CASE(create_sparse_problem) {
  uint nx = 3, nu = 2, nc = 1;
  uint horz = 1;
  VectorXs x0;
  x0.setRandom(nx);
  problem_t problem = short_problem(x0, horz, nx, nu, nc);
  Eigen::SparseMatrix<double> kktMat;
  VectorXs kktRhs;
  gar::lqrCreateSparseMatrix<false>(problem, 1e-8, 1e-6, kktMat, kktRhs);

  auto test_equal = [&] {
    auto [kktDense, rhsDense] = gar::lqrDenseMatrix(problem, 1e-8, 1e-6);

    fmt::println("kktMatrix (sparse):\n{}", kktMat.toDense());
    fmt::println("kktMatrix (dense):\n{}", kktDense);
    fmt::println("kktRhs (sparse) {}", kktRhs.transpose());
    fmt::println("kktRhs (dense) {}", rhsDense.transpose());
    BOOST_CHECK(rhsDense.isApprox(kktRhs));
    BOOST_CHECK(kktDense.isApprox(kktMat.toDense()));
  };

  test_equal();

  BOOST_TEST_MESSAGE("Test after modifying problem parameters.");
  // modify problem
  problem.g0.setRandom();
  problem.stages[0].Q = sampleWishartDistributedMatrix(nx, nx + 1);
  problem.stages[0].R = sampleWishartDistributedMatrix(nx, nx + 1);
  // update
  gar::lqrCreateSparseMatrix<true>(problem, 1e-8, 1e-6, kktMat, kktRhs);

  test_equal();
}

BOOST_AUTO_TEST_CASE(cholmod_short_horz) {
  const double mu = 1e-14;
  uint nx = 2, nu = 2;
  uint horz = 10;
  VectorXs x0;
  x0.setRandom(nx);
  problem_t problem = generate_problem(x0, horz, nx, nu);
  gar::CholmodLqSolver<double> solver{problem};

  auto [xs, us, vs, lbdas] = gar::lqrInitializeSolution(problem);

  // solver.backward(mu, mu);
  // solver.forward(xs, us, vs, lbdas);

  // auto [dynErr, cstErr, dualErr] = gar::lqrComputeKktError(
  //     problem, xs, us, vs, lbdas, mu, mu, std::nullopt, true);
  // fmt::print("KKT errors: d = {:.4e} / dual = {:.4e}\n", dynErr, dualErr);
}
