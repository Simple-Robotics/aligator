add_library(gar_test_utils STATIC util.cpp)
target_link_libraries(gar_test_utils PRIVATE aligator::gar)
set_target_properties(gar_test_utils PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "../")

set(_gar_test_deps aligator::gar gar_test_utils)

function(add_gar_test name)
  _add_test_prototype(${name} "gar-" "${_gar_test_deps}")
endfunction(add_gar_test)

add_gar_test(riccati)
add_gar_test(parallel proxddp)
