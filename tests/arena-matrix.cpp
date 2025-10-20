#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "aligator/core/arena-matrix.hpp"

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

TEST_CASE("managed_matrix_create_copy") {
  AllocMatrixType a{10, 10, g_alloc};
  REQUIRE(a.get_allocator() == g_alloc);
  REQUIRE(a.size() == 100);
  REQUIRE(a.allocatedSize() == a.size());

  AllocMatrixType b{a};
  // copy ctor does not propagate allocator
  REQUIRE(a.get_allocator() != b.get_allocator());
  REQUIRE(a.data() != b.data());

  // copy ctor using original allocator
  AllocMatrixType c{a, g_alloc};
  REQUIRE(a.get_allocator() == c.get_allocator());
  REQUIRE(a.data() != c.data());
}

TEST_CASE("managed_matrix_copy_asignment") {
  AllocMatrixType a{10, 10, g_alloc};
  a.setOnes();
  AllocMatrixType d = a;
  REQUIRE(a.get_allocator() != d.get_allocator());
  REQUIRE(a.isApprox(d));

  AllocMatrixType c{g_alloc};
  REQUIRE(a.get_allocator() == c.get_allocator());
  c = a;
  REQUIRE(a.isApprox(c));

  std::cout << "c matrix:\n" << c << std::endl;
  c.setIdentity();
  std::cout << "c = Id matrix:\n" << c << std::endl;
  c += a;
  std::cout << "c = Id + a matrix:\n" << c << std::endl;

  c = a + a;
  REQUIRE(c.isApprox(a + a));
  std::cout << "c = a + a matrix:\n" << c << std::endl;
  c.noalias() = a * a;
  std::cout << "c = a * a matrix:\n" << c << std::endl;

  // interop with vanilla Eigen::Matrix
  MatrixXd b;
  b.setRandom(10, 10);
  c.noalias() = a * b;
  REQUIRE(c.isApprox(a * b));

  c.topRows(4).setZero();
  std::cout << "c (top 4 is 0):\n" << c << std::endl;

  a.setZero(4, 4);
  std::cout << "a.setZero(4, 4):\n" << a << std::endl;
}

TEST_CASE("managed_matrix_resize") {
  AllocMatrixType a{4, 6, g_alloc};
  REQUIRE(a.rows() == 4);
  REQUIRE(a.cols() == 6);

  a.resize(4, 8);
  REQUIRE(a.rows() == 4);
  REQUIRE(a.cols() == 8);
  std::cout << "a (4, 8) (uninitialized):\n" << a << std::endl;

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

TEST_CASE("managed_matrix_move") {
  AllocMatrixType a{6, 10, g_alloc};
  auto *pdata = a.data();
  a.setRandom();
  REQUIRE(a.rows() == 6);
  REQUIRE(a.cols() == 10);
  REQUIRE(a.get_allocator() == g_alloc);

  AllocMatrixType a_copy = a;

  AllocMatrixType b{std::move(a)};
  REQUIRE(b.get_allocator() == g_alloc);
  REQUIRE(b.data() == pdata);
  REQUIRE(b == a_copy);

  AllocMatrixType c;
  c = std::move(b);
  REQUIRE(c.get_allocator() == polymorphic_allocator{});
  REQUIRE(c.data() != pdata);
  REQUIRE(c == a_copy);
}
