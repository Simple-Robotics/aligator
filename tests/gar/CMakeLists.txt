set(_gar_test_deps aligator::gar gar_test_utils)

function(add_gar_test name)
  _add_test_prototype(${name} "gar-" "${_gar_test_deps}")
endfunction(add_gar_test)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  add_gar_test(cholmod)
endif()
add_gar_test(riccati)
add_gar_test(block-matrix)
if(BUILD_WITH_OPENMP_SUPPORT)
  add_gar_test(parallel aligator)
endif()
