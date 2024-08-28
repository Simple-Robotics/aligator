/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#include <boost/test/unit_test.hpp>

#include "aligator/context.hpp"
#include "aligator/gar/blk-matrix.hpp"
#include "aligator/gar/block-tridiagonal.hpp"
#include <Eigen/Cholesky>

#include "test_util.hpp"
#include <proxsuite-nlp/fmt-eigen.hpp>

using namespace aligator;
using namespace aligator::context;

using MatrixXs = math_types<Scalar>::MatrixXs;
using MatrixRef = Eigen::Ref<MatrixXs>;
using VectorXs = math_types<Scalar>::VectorXs;

BOOST_AUTO_TEST_CASE(blk22) {
  std::array<long, 2> dims = {4, 6};
  BlkMatrix<MatrixXs, 2, 2> blk(dims, dims);
  blk.setZero();
  blk(0, 0).setOnes();
  blk(1, 0).setRandom();

  fmt::print("mat:\n{}\n", blk.matrix());
  BOOST_CHECK_EQUAL(blk.rows(), 10);
  BOOST_CHECK_EQUAL(blk.cols(), 10);

  BlkMatrix<MatrixRef, 1, 2> b12 = blk.topBlkRows<1>();
  fmt::print("b12:\n{}\n", b12.matrix());

  BOOST_CHECK_EQUAL(b12.rows(), 4);
  BOOST_CHECK_EQUAL(b12.cols(), 10);
}

BOOST_AUTO_TEST_CASE(dynamicblkvec) {
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
  BOOST_CHECK_EQUAL(bvtop2.rows(), 7);
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

  BTAG_Fixture() : diagonal(N + 1), sup(N), sub(N), facs() {
    for (size_t i = 0; i <= N; i++) {
      diagonal[i] = sampleWishartDistributedMatrix(nx, nx + 1);
    }

    std::fill_n(sup.begin(), N, B);
    std::fill_n(sub.begin(), N, B.transpose());

    std::vector<long> dims(N + 1);
    std::fill_n(dims.begin(), N + 1, nx);
    BOOST_CHECK(dims.size() == N + 1);
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

BOOST_FIXTURE_TEST_CASE(block_tridiag_solve_up_looking, BTAG_Fixture) {

  bool ret = gar::symmetricBlockTridiagSolve(sub, diagonal, sup, vec, facs);
  BOOST_CHECK(ret);

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
    BOOST_CHECK(vec.matrix().isApprox(rhs.matrix(), 1e-12));
  }
}

BOOST_FIXTURE_TEST_CASE(block_tridiag_solve_down_looking, BTAG_Fixture) {

  bool ret =
      gar::symmetricBlockTridiagSolveDownLooking(sub, diagonal, sup, vec, facs);
  BOOST_CHECK(ret);

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
    BOOST_CHECK(vec.matrix().isApprox(rhs.matrix(), 1e-12));
  }
}
