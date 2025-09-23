set(TEST_NAMES riccati block-matrix)

if(BUILD_WITH_OPENMP_SUPPORT)
  list(APPEND TEST_NAMES parallel)
endif()
if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(APPEND TEST_NAMES cholmod)
endif()

foreach(test ${TEST_NAMES})
  add_aligator_test(${test} gar_test_utils)
endforeach()

add_aligator_test2(knot gar_test_utils)
