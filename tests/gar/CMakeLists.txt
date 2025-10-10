set(TEST_NAMES riccati block-matrix knot)

if(BUILD_WITH_OPENMP_SUPPORT)
  list(APPEND TEST_NAMES parallel)
endif()

foreach(test ${TEST_NAMES})
  add_aligator_test2(${test} gar_test_utils)
endforeach()
