#include "aligator/gar/eigen-pmr.hpp"
#include <iostream>

#include <boost/test/unit_test.hpp>

using namespace aligator;

BOOST_AUTO_TEST_CASE(eigen_pmr_matrix) {
  auto &res = *std::pmr::get_default_resource();
  polymorphic_allocator alloc{&res};
  EigenPmr<double> m{4, 4, alloc};
  BOOST_CHECK(m.data() != NULL);
  BOOST_CHECK_EQUAL(m.rows(), 4);
  BOOST_CHECK_EQUAL(m.cols(), 4);
  //
  m.setOnes();
  std::cout << m << std::endl;
  m = 2 * m;
  std::cout << m << std::endl;
}

BOOST_AUTO_TEST_CASE(eigen_pmr_vector) {
  auto &res = *std::pmr::get_default_resource();
  polymorphic_allocator alloc{&res};
  EigenPmr<double, -1, 1> v{5, 1, alloc};
  BOOST_CHECK(v.data() != NULL);
  BOOST_CHECK_EQUAL(v.rows(), 5);
  BOOST_CHECK_EQUAL(v.cols(), 1);
  //
  v.setOnes();
  v(0) = 42.3;
  std::cout << v << std::endl;

  EigenPmr<double, -1, -1> m{2, 5, alloc};
  m.setOnes();
  std::cout << m << std::endl;
  EigenPmr<double, -1, 1> p{2, alloc};
  p.noalias() = m * v;
  std::cout << p << std::endl;

  auto s = std::move(p);
  std::cout << "copy s:\n" << s << std::endl;
  s(0) = -10;
  std::cout << "modified s:\n" << s << std::endl;
}
