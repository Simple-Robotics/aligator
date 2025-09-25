#include "aligator/gar/proximal-riccati.hpp"
#include "aligator/gar/cholmod-solver.hpp"
#include "aligator/gar/utils.hpp"

#include <aligator/fmt-eigen.hpp>

#include <catch2/catch_test_macros.hpp>
#include "test_util.hpp"

using namespace aligator;

TEST_CASE("helper_assignment_dense", "[gar]") {
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
  gar::helpers::sparseAssignDenseBlock(10, 11, submat, mat, false);
  REQUIRE(densemat.isApprox(mat.toDense()));
  fmt::println("submat 1:\n{}", submat);

  submat.setRandom(5, 5);
  densemat.block(3, 4, 5, 5) = submat;
  gar::helpers::sparseAssignDenseBlock(3, 4, submat, mat, false);
  REQUIRE(densemat.isApprox(mat.toDense()));
  fmt::println("submat 2:\n{}", submat);

  fmt::println("dense mat:\n{}", densemat);
  fmt::println("sparse mat:\n{}", mat.toDense());

  // reinitialize matrix

  densemat.setZero();
  mat.setZero();
  densemat.block(3, 4, 5, 5) = submat;
  gar::helpers::sparseAssignDenseBlock(3, 4, submat, mat, true);
  REQUIRE(densemat.isApprox(mat.toDense()));
}

problem_t short_problem(Eigen::Ref<const VectorXs> x0, uint horz, uint nx,
                        uint nu, uint nc) {

  problem_t::KnotVector knots;
  knots.reserve(horz + 1);
  const uint nc0 = (uint)x0.size();

  for (uint t = 0; t <= horz; t++) {
    problem_t::KnotType &knot = knots.emplace_back(nx, nu, nc);
    knot.Q.setConstant(+0.11);
    knot.R.setConstant(+0.12);
    knot.S.setConstant(-0.2);
    knot.q.setConstant(13.);
    knot.r.setConstant(-13);

    knot.C.setConstant(-2.);
    knot.D.setConstant(-3.);
    knot.d.setConstant(-4.);

    knot.f.setConstant(-0.3);
    knot.A.setConstant(0.2);
    knot.B.setConstant(0.3);
    knot.E.setIdentity();
    knot.E *= -1;
  }
  problem_t out{std::move(knots), nc0};
  out.G0.setConstant(-3.14);
  out.g0.setConstant(42);
  return out;
}

TEST_CASE("create_sparse_problem", "[gar]") {
  uint nx = 3, nu = 2, nc = 1;
  uint horz = 1;
  VectorXs x0;
  x0.setRandom(nx);
  const double mueq = 1e-6;
  problem_t problem = short_problem(x0, horz, nx, nu, nc);
  Eigen::SparseMatrix<double> kktMat;
  VectorXs kktRhs;
  gar::lqrCreateSparseMatrix(problem, mueq, kktMat, kktRhs, false);

  auto test_equal = [&] {
    auto [kktDense, rhsDense] = gar::lqrDenseMatrix(problem, 1e-8, 1e-6);

    fmt::println("kktMatrix (sparse):\n{}", kktMat.toDense());
    fmt::println("kktMatrix (dense):\n{}", kktDense);
    fmt::println("kktRhs (sparse) {}", kktRhs.transpose());
    fmt::println("kktRhs (dense)  {}", rhsDense.transpose());
    REQUIRE(rhsDense.isApprox(kktRhs));
    REQUIRE(kktDense.isApprox(kktMat.toDense()));
  };

  test_equal();

  // modify problem
  problem.g0.setRandom();
  problem.stages[0].Q = sampleWishartDistributedMatrix(nx, nx + 1);
  problem.stages[0].R = sampleWishartDistributedMatrix(nu, nu + 1);
  // update
  gar::lqrCreateSparseMatrix(problem, mueq, kktMat, kktRhs, true);

  test_equal();
}

TEST_CASE("cholmod_short_horz", "[gar]") {
  const double mueq = 1e-10;
  uint nx = 4, nu = 4;
  uint horz = 10;
  constexpr double TOL = 1e-11;
  VectorXs x0 = VectorXs::Random(nx);
  problem_t problem = generate_problem(x0, horz, nx, nu);
  gar::CholmodLqSolver<double> solver{problem, 1};

  auto [xs, us, vs, lbdas] = gar::lqrInitializeSolution(problem);

  bool ret = solver.backward(mueq);
  REQUIRE(ret);
  {
    // test here because backward(mudyn, mueq) sets right values of mu
    auto [denseKkt, rhsDense] = gar::lqrDenseMatrix(problem, mueq, mueq);
    REQUIRE(denseKkt.isApprox(solver.kktMatrix.toDense()));
    REQUIRE(rhsDense.isApprox(solver.kktRhs));
  }
  solver.forward(xs, us, vs, lbdas);

  const double sparse_residual = solver.computeSparseResidual();
  fmt::println("Sparse solver residual: {:.4e}", sparse_residual);
  REQUIRE(sparse_residual <= TOL);

  REQUIRE(ret);

  auto [dynErr, cstErr, dualErr, maxErr] =
      computeKktError(problem, xs, us, vs, lbdas, std::nullopt, mueq, true);
  fmt::print("KKT errors: d = {:.4e} / dual = {:.4e}\n", dynErr, dualErr);
  REQUIRE(maxErr <= TOL);

  {
    gar::ProximalRiccatiSolver<double> solver2{problem};
    ret = solver2.backward(mueq);
    REQUIRE(ret);
    auto [xs2, us2, vs2, lbdas2] = gar::lqrInitializeSolution(problem);
    solver2.forward(xs2, us2, vs2, lbdas2);

    auto [dynErr2, cstErr2, dualErr2, maxErr] = computeKktError(
        problem, xs2, us2, vs2, lbdas2, std::nullopt, mueq, true);
    fmt::print("KKT errors: d = {:.4e} / dual = {:.4e}\n", dynErr2, dualErr2);

    for (uint i = 0; i <= horz; i++) {
      fmt::println("xerr[{:d}] = {:.3e}", i, math::infty_norm(xs[i] - xs2[i]));
      fmt::println("lerr[{:d}] = {:.3e}", i,
                   math::infty_norm(lbdas[i] - lbdas2[i]));
    }
  }
}
