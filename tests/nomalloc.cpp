/// @file
/// @brief Tests for the no-malloc macros

#include <boost/test/unit_test.hpp>
#include <aligator/math.hpp>

BOOST_AUTO_TEST_CASE(begin_end_basic) {
  ALIGATOR_NOMALLOC_BEGIN;
  BOOST_CHECK(!Eigen::internal::is_malloc_allowed());
  ALIGATOR_NOMALLOC_END;

  // check if malloc legal
  BOOST_CHECK(Eigen::internal::is_malloc_allowed());
}

BOOST_AUTO_TEST_CASE(scoped) {
  {
    ALIGATOR_NOMALLOC_SCOPED;
    BOOST_CHECK(!Eigen::internal::is_malloc_allowed());
  }

  BOOST_CHECK(aligator::internal::get_cached_malloc_status());
  BOOST_CHECK(Eigen::internal::is_malloc_allowed());
}

BOOST_AUTO_TEST_CASE(nested) {
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
}
