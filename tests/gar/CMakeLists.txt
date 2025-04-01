set(_gar_test_deps ${PROJECT_NAME} gar_test_utils)

function(add_gar_test name)
  _add_test_prototype(${name} "gar-" "${_gar_test_deps}")
endfunction(add_gar_test)

set(TEST_NAMES riccati block-matrix knot matrix-array)

if(BUILD_WITH_OPENMP_SUPPORT)
  list(APPEND TEST_NAMES parallel)
endif()
if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(APPEND TEST_NAMES cholmod)
endif()

foreach(test ${TEST_NAMES})
  add_gar_test(${test})
endforeach()
