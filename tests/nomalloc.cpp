/// @file
/// @brief Tests for the no-malloc macros

#include <catch2/catch_test_macros.hpp>
#include "aligator/math.hpp"

TEST_CASE("begin_end_basic") {
  ALIGATOR_NOMALLOC_BEGIN;
  REQUIRE_FALSE(Eigen::internal::is_malloc_allowed());
  ALIGATOR_NOMALLOC_END;

  // check if malloc legal
  REQUIRE(Eigen::internal::is_malloc_allowed());
}

TEST_CASE("scoped") {
  {
    ALIGATOR_NOMALLOC_SCOPED;
    REQUIRE_FALSE(Eigen::internal::is_malloc_allowed());
  }

  REQUIRE(aligator::internal::get_cached_malloc_status());
  REQUIRE(Eigen::internal::is_malloc_allowed());
}

TEST_CASE("nested") {
  ALIGATOR_NOMALLOC_BEGIN;
  {
    ALIGATOR_NOMALLOC_SCOPED;
    REQUIRE_FALSE(aligator::internal::get_cached_malloc_status());
    REQUIRE_FALSE(Eigen::internal::is_malloc_allowed());
    {
      ALIGATOR_NOMALLOC_SCOPED;
      REQUIRE_FALSE(Eigen::internal::is_malloc_allowed());
      ALIGATOR_NOMALLOC_END; // allow Eigen's malloc, do not modify cached value
      REQUIRE(Eigen::internal::is_malloc_allowed());
      REQUIRE_FALSE(aligator::internal::get_cached_malloc_status());
      // exiting restores the malloc status
    }
  }
  REQUIRE_FALSE(Eigen::internal::is_malloc_allowed());
  ALIGATOR_NOMALLOC_END;
  REQUIRE(Eigen::internal::is_malloc_allowed());
}
