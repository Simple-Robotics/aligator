#
# Copyright (C) 2023 LAAS-CNRS, INRIA
#

file(
  GLOB PYTHON_TESTS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS
  test_*.py
)
if(NOT BUILD_CROCODDYL_COMPAT)
  list(REMOVE_ITEM PYTHON_TESTS test_compat_croc.py)
endif()
if(NOT BUILD_WITH_PINOCCHIO_SUPPORT)
  list(REMOVE_ITEM PYTHON_TESTS test_center_of_mass.py)
  list(REMOVE_ITEM PYTHON_TESTS test_frames.py)
  list(REMOVE_ITEM PYTHON_TESTS test_rollout.py)
endif()
if(NOT PINOCCHIO_V3)
  list(REMOVE_ITEM PYTHON_TESTS test_constrained_dynamics.py)
endif()
if(NOT PROXSUITE_NLP_WITH_PINOCCHIO_SUPPORT)
  list(REMOVE_ITEM PYTHON_TESTS test_kinodynamics.py)
endif()
make_directory(${CMAKE_CURRENT_BINARY_DIR})

foreach(pyfile ${PYTHON_TESTS})
  get_filename_component(test_name ${pyfile} NAME_WE)
  string(
    REGEX REPLACE
    "^test_"
    "${PROJECT_NAME}-test-py-"
    test_name
    ${test_name}
  )
  string(REPLACE "_" "-" test_name ${test_name})
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/${pyfile}
      ${CMAKE_CURRENT_BINARY_DIR}/${pyfile}
  )
  message(STATUS "Adding Python test: ${test_name}")
  ADD_PYTHON_UNIT_TEST(
    ${test_name}
    "tests/python/${pyfile}"
    "tests/python"
    "bindings/python"
  )
  set_tests_properties("${test_name}" PROPERTIES DEPENDS ${PYLIB_NAME})
endforeach()

function(add_test_binding_lib test_name)
  CREATE_CTEST_BUILD_TESTS_TARGET()

  set(target_name "${PROJECT_NAME}-${test_name}")
  Python3_add_library(${target_name} MODULE "${test_name}.cpp")
  set_standard_output_directory(${target_name})
  target_link_libraries(${target_name} PUBLIC ${PYLIB_NAME})
  target_link_libraries(${target_name} PUBLIC eigenpy::eigenpy)
  set_target_properties(
    ${target_name}
    PROPERTIES
      PREFIX ""
      SUFFIX ${PYTHON_EXT_SUFFIX}
      LIBRARY_OUTPUT_NAME ${test_name}
      RUNTIME_OUTPUT_NAME ${test_name}
  )
  # test for possible import segfault
  add_test(
    NAME ${target_name}
    COMMAND ${PYTHON_EXECUTABLE} -c "import ${test_name}"
  )
  message(STATUS "Adding Python/C++ test ${test_name}")

  add_dependencies(build_tests ${target_name})
  if(NOT BUILD_TESTING)
    set_tests_properties(${target_name} PROPERTIES DEPENDS ctest_build_tests)
  endif()
endfunction()
