/// @file
/// @brief Tests for the no-malloc macros

#include <boost/test/unit_test.hpp>
#include <aligator/math.hpp>
#include <aligator/eigen-macros.hpp>

BOOST_AUTO_TEST_CASE(begin_end_basic) {
#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
  ALIGATOR_NOMALLOC_BEGIN;
  BOOST_CHECK(!Eigen::internal::is_malloc_allowed());
  ALIGATOR_NOMALLOC_END;

  // check if malloc legal
  BOOST_CHECK(Eigen::internal::is_malloc_allowed());
#endif
}

BOOST_AUTO_TEST_CASE(scoped) {
#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
  {
    ALIGATOR_NOMALLOC_SCOPED;
    BOOST_CHECK(!Eigen::internal::is_malloc_allowed());
  }

  BOOST_CHECK(aligator::internal::get_cached_malloc_status());
  BOOST_CHECK(Eigen::internal::is_malloc_allowed());
#endif
}

BOOST_AUTO_TEST_CASE(nested) {
#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
  ALIGATOR_NOMALLOC_BEGIN;
  {
    ALIGATOR_NOMALLOC_SCOPED;
    BOOST_CHECK(!aligator::internal::get_cached_malloc_status());
    BOOST_CHECK(!Eigen::internal::is_malloc_allowed());
    {
      ALIGATOR_NOMALLOC_SCOPED;
      BOOST_CHECK(!Eigen::internal::is_malloc_allowed());
      ALIGATOR_NOMALLOC_END; // allow Eigen's malloc, do not modify cached value
      BOOST_CHECK(Eigen::internal::is_malloc_allowed());
      BOOST_CHECK(!aligator::internal::get_cached_malloc_status());
      // exiting restores the malloc status
    }
  }
  BOOST_CHECK(!Eigen::internal::is_malloc_allowed());
  ALIGATOR_NOMALLOC_END;
  BOOST_CHECK(Eigen::internal::is_malloc_allowed());
#endif
}
