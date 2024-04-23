set(_gar_test_deps aligator::gar gar_test_utils)

function(add_gar_test name)
  _add_test_prototype(${name} "gar-" "${_gar_test_deps}")
endfunction(add_gar_test)

add_gar_test(riccati)
add_gar_test(block-matrix)
add_gar_test(parallel aligator)

add_executable(run-parallel run-parallel.cpp)
target_link_libraries(run-parallel PRIVATE ${_gar_test_deps})
