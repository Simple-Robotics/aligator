#include <boost/test/unit_test.hpp>

#include "aligator/context.hpp"
#include "aligator/gar/blk-matrix.hpp"

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
