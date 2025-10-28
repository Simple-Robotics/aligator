/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
/// @author Wilson Jallet
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_get_random_seed.hpp>

#include "aligator/context.hpp"
#include "aligator/core/blk-matrix.hpp"
#include "aligator/gar/block-tridiagonal.hpp"
#include "aligator/core/bunchkaufman.hpp"
#include <Eigen/Cholesky>

#include "test_util.hpp"
#include <aligator/fmt-eigen.hpp>

using namespace aligator;
using context::MatrixRef;
using context::MatrixXs;
using context::VectorXs;
using BlkVec = BlkMatrix<VectorXs, -1, 1>;

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
  std::mt19937 rng{Catch::getSeed()};
  normal_unary_op normal_op{rng};
  size_t N = 6;
  uint nx = 2;
  MatrixXs B = MatrixXs::NullaryExpr(nx, nx, normal_op);

  std::vector<MatrixXs> diagonal;
  std::vector<MatrixXs> sup;
  std::vector<MatrixXs> sub;
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

constexpr double prec = 1e-12;

TEST_CASE_METHOD(BTAG_Fixture, "block_tridiag_solve_up_looking", "[gar]") {

  bool ret = gar::symmetricBlockTridiagSolve(sub, diagonal, sup, vec, facs);
  REQUIRE(ret);

  aligator::BunchKaufman<MatrixXs> ldlt(densemat);
  ldlt.solveInPlace(rhs.matrix());

  REQUIRE(vec.isApprox(rhs, prec));
}

TEST_CASE_METHOD(BTAG_Fixture, "block_tridiag_solve_down_looking", "[gar]") {

  bool ret =
      gar::symmetricBlockTridiagSolveDownLooking(sub, diagonal, sup, vec, facs);
  REQUIRE(ret);

  aligator::BunchKaufman<MatrixXs> ldlt(densemat);
  ldlt.solveInPlace(rhs.matrix());

  REQUIRE(vec.isApprox(rhs, prec));
}
