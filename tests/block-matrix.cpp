#include <boost/test/unit_test.hpp>

#include "proxddp/context.hpp"
#include "proxddp/gar/BlkMatrix.hpp"

using namespace proxddp;
using namespace proxddp::context;

using MatrixXs = math_types<Scalar>::MatrixXs;
using VectorXs = math_types<Scalar>::VectorXs;

BOOST_AUTO_TEST_CASE(blk22) {
  std::array<long, 2> dims = {4, 6};
  BlkMatrix<MatrixXs, 2, 2> blk(dims);
  blk.setZero();
  blk(0, 0).setOnes();
  blk(1, 0).setRandom();

  fmt::print("mat:\n{}\n", blk.data);
}

BOOST_AUTO_TEST_CASE(dynamicblkvec) {
  std::vector<long> dims{2, 5};
  BlkMatrix<VectorXs, -1, 1> vec(dims);
  vec.setZero();
  vec.blockSegment(0).setRandom();
  vec.blockSegment(1).setConstant(42.);
  fmt::print("rowDims: {}\n", vec.rowDims());
  fmt::print("colDims: {}\n", vec.colDims());
  fmt::print("rowIdx: {}\n", vec.rowIndices());
  fmt::print("colIdx: {}\n", vec.colIndices());

  fmt::print("vec:\n{}\n", vec.data);
}
