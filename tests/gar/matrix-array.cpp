#include <boost/test/unit_test.hpp>

#include "aligator/context.hpp"
#include "aligator/gar/matrix-array.hpp"

using namespace aligator;
using context::MatrixXs;
using context::VectorXs;

BOOST_AUTO_TEST_CASE(vecarr) {
  polymorphic_allocator alloc{};
  std::pmr::vector<long> dims{2, 4};
  VectorArray<double> arr{dims, alloc};
  VectorXs v0, v1;
  v0.setRandom(2);
  v1.setOnes(4);
  arr[0] = v0;
  arr[1] = v1;

  BOOST_CHECK(arr[0] == v0);
  BOOST_CHECK(arr[1] == v1);

  // move
  auto arr2 = std::move(arr);
  BOOST_CHECK(arr[0] == v0);
  BOOST_CHECK(arr[1] == v1);
}

BOOST_AUTO_TEST_CASE(matarr) {
  polymorphic_allocator alloc{};
  std::pmr::vector<long> rows{2, 4};
  std::pmr::vector<long> cols{3, 4};
  MatrixArray<double, false> arr{rows, cols, alloc};
  MatrixXs M;
  M.setRandom(2, 3);
  arr[0] = M;
  arr[1].setIdentity();

  BOOST_CHECK(arr[0] == M);
  BOOST_CHECK(arr[1] == MatrixXs::Identity(4, 4));

  // move
  auto arr2 = std::move(arr);
  BOOST_CHECK(arr[0] == M);
  BOOST_CHECK(arr[1] == MatrixXs::Identity(4, 4));
}
