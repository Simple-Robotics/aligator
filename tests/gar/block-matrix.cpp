#include <boost/test/unit_test.hpp>

#include "aligator/context.hpp"
#include "aligator/gar/blk-matrix.hpp"
#include "aligator/gar/block-tridiagonal-solver.hpp"

#include "test_util.hpp"

using namespace aligator;
using namespace aligator::context;
namespace utf = boost::unit_test::framework;

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

BOOST_AUTO_TEST_CASE(block_tridiag_solve) {
  auto test_case_name = utf::current_test_case().full_name();
  BOOST_TEST_MESSAGE(test_case_name);
  size_t N = 6;
  uint nx = 2;
  MatrixXs B = MatrixXs::NullaryExpr(nx, nx, normal_unary_op{});

  std::vector<MatrixXs> diagonal(N + 1);
  std::vector<MatrixXs> sup(N);
  std::vector<MatrixXs> sub(N);

  for (size_t i = 0; i <= N; i++) {
    diagonal[i] = wishart_dist_matrix(nx, nx + 1);
  }

  std::fill_n(sup.begin(), N, B);
  std::fill_n(sub.begin(), N, B.transpose());

  BOOST_CHECK(gar::internal::check_block_tridiag(sub, diagonal, sup));

  using BlkVec = BlkMatrix<VectorXs, -1, 1>;
  std::vector<long> dims(N + 1);
  std::fill_n(dims.begin(), N + 1, nx);
  BOOST_CHECK(dims.size() == N + 1);
  BlkVec vec(dims);
  vec.matrix().setOnes();

  MatrixXs densemat = block_tridiag_to_dense(sub, diagonal, sup);
  fmt::print("Dense problem matrix:\n{}\n", densemat);
  BlkVec densevec = vec;

  gar::symmetric_block_tridiagonal_solve(sub, diagonal, sup, vec);

  for (size_t i = 0; i <= N; i++) {
    fmt::print("rhs[{:d}] = {}\n", i, vec[i].transpose());
  }

  {
    // alternative solve
    Eigen::LDLT<MatrixXs> ldlt(densemat);
    ldlt.solveInPlace(densevec.matrix());

    fmt::print("Alternative solve:\n");
    for (size_t i = 0; i <= N; i++) {
      fmt::print("rhs[{:d}] = {}\n", i, densevec[i].transpose());
    }
    BOOST_CHECK(vec.matrix().isApprox(densevec.matrix(), 1e-12));
  }
}
