#include <Eigen/Core>
#include <boost/test/unit_test.hpp>

#include "aligator/gar/memory-allocator.hpp"

static aligator::polymorphic_allocator alloc{};
using Eigen::MatrixXd;
using MapType = MatrixXd::AlignedMapType;

MapType allocate_eigen_matrix(Eigen::Index n, Eigen::Index m) {
  size_t size = size_t(n * m);
  double *data = alloc.allocate<double>(size);
  return MapType{data, n, m};
}

struct map_owning_type {
  MapType view;

  ~map_owning_type() {
    alloc.deallocate<double>(view.data(), size_t(view.size()));
    view.~MapType();
  }
};

BOOST_AUTO_TEST_CASE(struct_creating_matrix) {
  map_owning_type a{allocate_eigen_matrix(2, 2)};
  a.view.setOnes();

  BOOST_CHECK(a.view == MatrixXd::Ones(2, 2));

  Eigen::Matrix2d b;
  b.setRandom();

  a.view = b;
  BOOST_CHECK_EQUAL(a.view, b);
}
