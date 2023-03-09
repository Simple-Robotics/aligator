#
# Copyright (C) 2023 LAAS-CNRS, INRIA
#

file(
  GLOB PYTHON_TESTS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  test_*.py)
if(NOT BUILD_CROCODDYL_COMPAT)
  list(REMOVE_ITEM PYTHON_TESTS test_compat_croc.py)
endif()
if(NOT BUILD_WITH_PINOCCHIO_PYTHON_BINDINGS)
  list(REMOVE_ITEM PYTHON_TESTS test_constrained_dynamics.py)
  list(REMOVE_ITEM PYTHON_TESTS test_frames.py)
endif()
make_directory(${CMAKE_CURRENT_BINARY_DIR})

foreach(pyfile ${PYTHON_TESTS})
  get_filename_component(pyfile_name ${pyfile} NAME)
  message(STATUS "Adding Python test: ${pyfile_name}")
  add_python_unit_test(${pyfile_name} "tests/python/${pyfile}" "tests/python")
  set_tests_properties("${pyfile_name}" PROPERTIES DEPENDS ${PYLIB_NAME})
endforeach()

function(add_test_binding_lib test_name)
  create_ctest_build_tests_target()

  add_library(${test_name} SHARED "${test_name}.cpp")
  target_link_libraries(${test_name} PUBLIC ${PYLIB_NAME})
  set_target_properties(${test_name} PROPERTIES PREFIX "" SUFFIX ${PYTHON_EXT_SUFFIX})
  # test for possible import segfault
  add_test(NAME ${test_name} COMMAND ${PYTHON_EXECUTABLE} -c "import ${test_name}")
  message(STATUS "Adding Python/C++ test ${test_name}")

  add_dependencies(build_tests ${test_name})
  if(NOT BUILD_TESTING)
    set_tests_properties(${test_name} PROPERTIES DEPENDS ctest_build_tests)
  endif()
endfunction()

add_test_binding_lib(bind_optional)
