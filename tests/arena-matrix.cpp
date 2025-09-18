#include <boost/test/unit_test.hpp>

#include "aligator/memory/arena-matrix.hpp"

using Scalar = double;
using aligator::polymorphic_allocator;
using Eigen::MatrixXd;
using AllocMatrixType = aligator::ArenaMatrix<MatrixXd>;
using MapType = AllocMatrixType::Base;
static_assert(aligator::is_eigen_v<AllocMatrixType>);
static_assert(std::uses_allocator_v<AllocMatrixType, polymorphic_allocator>);
static_assert(std::is_same_v<AllocMatrixType::PlainObject, MatrixXd>);
static_assert(std::is_same_v<MapType, MatrixXd::AlignedMapType>);

using alloc_traits = std::allocator_traits<polymorphic_allocator>;
static_assert(!alloc_traits::propagate_on_container_copy_assignment());
static_assert(!alloc_traits::propagate_on_container_move_assignment());
static_assert(!alloc_traits::propagate_on_container_swap());

std::pmr::monotonic_buffer_resource g_res{40000};
polymorphic_allocator g_alloc{&g_res};

BOOST_AUTO_TEST_SUITE(managed_matrix)

BOOST_AUTO_TEST_CASE(managed_matrix_create_copy) {
  AllocMatrixType a{10, 10, g_alloc};
  BOOST_CHECK(a.get_allocator() == g_alloc);
  BOOST_CHECK(a.size() == 100);
  BOOST_CHECK(a.allocatedSize() == a.size());

  AllocMatrixType b{a};
  // copy ctor does not propagate allocator
  BOOST_CHECK(a.get_allocator() != b.get_allocator());
  BOOST_CHECK(a.data() != b.data());

  // copy ctor using original allocator
  AllocMatrixType c{a, g_alloc};
  BOOST_CHECK(a.get_allocator() == c.get_allocator());
  BOOST_CHECK(a.data() != c.data());
}

BOOST_AUTO_TEST_CASE(managed_matrix_copy_asignment) {
  AllocMatrixType a{10, 10, g_alloc};
  a.setOnes();
  AllocMatrixType d = a;
  BOOST_CHECK(a.get_allocator() != d.get_allocator());
  BOOST_CHECK(a.isApprox(d));

  AllocMatrixType c{g_alloc};
  BOOST_CHECK(a.get_allocator() == c.get_allocator());
  c = a;
  BOOST_CHECK(a.isApprox(c));

  std::cout << "c matrix:\n" << c << std::endl;
  c.setIdentity();
  std::cout << "c = Id matrix:\n" << c << std::endl;
  c += a;
  std::cout << "c = Id + a matrix:\n" << c << std::endl;

  c = a + a;
  BOOST_CHECK(c.isApprox(a + a));
  std::cout << "c = a + a matrix:\n" << c << std::endl;
  c.noalias() = a * a;
  std::cout << "c = a * a matrix:\n" << c << std::endl;

  // interop with vanilla Eigen::Matrix
  MatrixXd b;
  b.setRandom(10, 10);
  c.noalias() = a * b;
  BOOST_CHECK(c.isApprox(a * b));

  c.topRows(4).setZero();
  std::cout << "c (top 4 is 0):\n" << c << std::endl;

  a.setZero(4, 4);
  std::cout << "a.setZero(4, 4):\n" << a << std::endl;
}

BOOST_AUTO_TEST_CASE(managed_matrix_resize) {
  AllocMatrixType a{4, 6, g_alloc};
  BOOST_CHECK_EQUAL(a.rows(), 4);
  BOOST_CHECK_EQUAL(a.cols(), 6);

  BOOST_TEST_MESSAGE("resize");
  a.resize(4, 8);
  BOOST_CHECK_EQUAL(a.rows(), 4);
  BOOST_CHECK_EQUAL(a.cols(), 8);
  std::cout << "a (4, 8) (uninitialized):\n" << a << std::endl;

  BOOST_TEST_MESSAGE("conservativeResize");
  a.conservativeResize(4, 3);
  a.setConstant(3.14);
  std::cout << "a.conservativeResize(4, 4):\n" << a << std::endl;
  a.conservativeResize(3, 6);
  std::cout << "a.conservativeResize(3, 6) (should have garbage):\n"
            << a << std::endl;
  a.conservativeResize(3, 9);
  std::cout << "a.conservativeResize(3, 9) (reallocates, should conserve "
               "existing values):\n"
            << a << std::endl;
}

BOOST_AUTO_TEST_CASE(managed_matrix_move) {
  AllocMatrixType a{6, 10, g_alloc};
  auto *pdata = a.data();
  a.setRandom();
  BOOST_CHECK_EQUAL(a.rows(), 6);
  BOOST_CHECK_EQUAL(a.cols(), 10);
  BOOST_CHECK(a.get_allocator() == g_alloc);

  AllocMatrixType a_copy = a;

  BOOST_TEST_MESSAGE("Move ctor");
  AllocMatrixType b{std::move(a)};
  BOOST_CHECK(b.get_allocator() == g_alloc);
  BOOST_CHECK(b.data() == pdata);
  BOOST_CHECK(b == a_copy);

  BOOST_TEST_MESSAGE("Move assignment op");
  AllocMatrixType c;
  c = std::move(b);
  BOOST_CHECK(c.get_allocator() == polymorphic_allocator{});
  BOOST_CHECK(c.data() != pdata);
  BOOST_CHECK(c == a_copy);
}

BOOST_AUTO_TEST_SUITE_END()
