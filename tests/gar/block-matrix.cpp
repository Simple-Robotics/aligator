/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#include <catch2/catch_test_macros.hpp>

#include "aligator/context.hpp"
#include "aligator/gar/blk-matrix.hpp"
#include "aligator/gar/block-tridiagonal.hpp"
#include <Eigen/Cholesky>

#include "test_util.hpp"
#include <aligator/fmt-eigen.hpp>

using namespace aligator;
using namespace aligator::context;

using MatrixXs = math_types<Scalar>::MatrixXs;
using MatrixRef = Eigen::Ref<MatrixXs>;
using VectorXs = math_types<Scalar>::VectorXs;

TEST_CASE("blk22", "[gar]") {
  std::array<long, 2> dims = {4, 6};
  BlkMatrix<MatrixXs, 2, 2> blk(dims, dims);
  blk.setZero();
  blk(0, 0).setOnes();
  blk(1, 0).setRandom();

  fmt::print("mat:\n{}\n", blk.matrix());
  REQUIRE(blk.rows() == 10);
  REQUIRE(blk.cols() == 10);

  BlkMatrix<MatrixRef, 1, 2> b12 = blk.topBlkRows<1>();
  fmt::print("b12:\n{}\n", b12.matrix());

  REQUIRE(b12.rows() == 4);
  REQUIRE(b12.cols() == 10);
}

TEST_CASE("dynamicblkvec", "[gar]") {
  std::vector<long> dims{2, 5, 2};
  BlkMatrix<VectorXs, -1, 1> vec(dims);
  vec.blockSegment(0).setRandom();
  vec[1].setConstant(42.);
  fmt::print("rowDims: {}\n", vec.rowDims());
  fmt::print("colDims: {}\n", vec.colDims());
  fmt::print("rowIdx: {}\n", vec.rowIndices());
  fmt::print("colIdx: {}\n", vec.colIndices());

  fmt::print("vec:\n{}\n", vec.matrix());

  BlkMatrix<Eigen::Ref<VectorXs>, -1, 1> bvtop2 = vec.topBlkRows(2);
  REQUIRE(bvtop2.rows() == 7);
}

struct BTAG_Fixture {
  size_t N = 6;
  uint nx = 2;
  MatrixXs B = MatrixXs::NullaryExpr(nx, nx, normal_unary_op{});

  std::vector<MatrixXs> diagonal;
  std::vector<MatrixXs> sup;
  std::vector<MatrixXs> sub;
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;
  BlkVec vec;
  MatrixXs densemat;
  BlkVec rhs;
  std::vector<Eigen::LDLT<MatrixXs>> facs;

  BTAG_Fixture()
      : diagonal(N + 1)
      , sup(N)
      , sub(N)
      , facs() {
    for (size_t i = 0; i <= N; i++) {
      diagonal[i] = sampleWishartDistributedMatrix(nx, nx + 1);
    }

    std::fill_n(sup.begin(), N, B);
    std::fill_n(sub.begin(), N, B.transpose());

    std::vector<long> dims(N + 1);
    std::fill_n(dims.begin(), N + 1, nx);
    REQUIRE(dims.size() == N + 1);
    vec = BlkVec(dims);
    vec.matrix().setOnes();

    densemat = gar::blockTridiagToDenseMatrix(sub, diagonal, sup);
    fmt::print("Dense problem matrix:\n{}\n", densemat);
    rhs = vec;
    for (size_t i = 0; i < diagonal.size(); i++) {
      facs.emplace_back(diagonal[i].cols());
    }
  }
};

TEST_CASE_METHOD(BTAG_Fixture, "block_tridiag_solve_up_looking", "[gar]") {

  bool ret = gar::symmetricBlockTridiagSolve(sub, diagonal, sup, vec, facs);
  REQUIRE(ret);

  for (size_t i = 0; i <= N; i++) {
    fmt::print("rhs[{:d}] = {}\n", i, vec[i].transpose());
  }

  {
    // alternative solve
    Eigen::LDLT<MatrixXs> ldlt(densemat);
    ldlt.solveInPlace(rhs.matrix());

    fmt::print("Alternative solve:\n");
    for (size_t i = 0; i <= N; i++) {
      fmt::print("rhs[{:d}] = {}\n", i, rhs[i].transpose());
    }
    REQUIRE(vec.matrix().isApprox(rhs.matrix(), 1e-12));
  }
}

TEST_CASE_METHOD(BTAG_Fixture, "block_tridiag_solve_down_looking", "[gar]") {

  bool ret =
      gar::symmetricBlockTridiagSolveDownLooking(sub, diagonal, sup, vec, facs);
  REQUIRE(ret);

  for (size_t i = 0; i <= N; i++) {
    fmt::print("rhs[{:d}] = {}\n", i, vec[i].transpose());
  }

  {
    // alternative solve
    Eigen::LDLT<MatrixXs> ldlt(densemat);
    ldlt.solveInPlace(rhs.matrix());

    fmt::print("Alternative solve:\n");
    for (size_t i = 0; i <= N; i++) {
      fmt::print("rhs[{:d}] = {}\n", i, rhs[i].transpose());
    }
    REQUIRE(vec.matrix().isApprox(rhs.matrix(), 1e-12));
  }
}
