#include <boost/test/unit_test.hpp>

#include "aligator/memory/managed-matrix.hpp"

using Scalar = double;
using aligator::polymorphic_allocator;
using AllocMatrixType =
    aligator::ManagedMatrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using MapType = AllocMatrixType::MapType;
static_assert(std::uses_allocator_v<AllocMatrixType, polymorphic_allocator>);
static_assert(std::is_same_v<AllocMatrixType::MatrixType, Eigen::MatrixXd>);
static_assert(std::is_same_v<MapType, Eigen::MatrixXd::AlignedMapType>);

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

  AllocMatrixType b{a};
  // copy ctor does not propagate allocator
  BOOST_CHECK(a.get_allocator() != b.get_allocator());

  // copy ctor using original allocator
  AllocMatrixType c{a, g_alloc};
  BOOST_CHECK(a.get_allocator() == c.get_allocator());
}

BOOST_AUTO_TEST_CASE(managed_matrix_copy_asignment) {
  AllocMatrixType a{10, 10, g_alloc};
  AllocMatrixType d;
  d = a;
  BOOST_CHECK(a.get_allocator() != d.get_allocator());
}

BOOST_AUTO_TEST_CASE(managed_matrix_create_move) {
  AllocMatrixType a{6, 10, g_alloc};

  AllocMatrixType b{std::move(a)};
  BOOST_CHECK(b.get_allocator() == g_alloc);

  // move-construct with a new allocator
  AllocMatrixType c{std::move(b), polymorphic_allocator{}};
  BOOST_CHECK(c.get_allocator() != g_alloc);
}

BOOST_AUTO_TEST_SUITE_END()
