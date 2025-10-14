#
# Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#

find_package(Catch2 3 REQUIRED)
include(Catch)

macro(ADD_TEST_CFLAGS test_name flag)
  set_property(
    TARGET ${test_name}
    APPEND_STRING
    PROPERTY COMPILE_FLAGS " ${flag}"
  )
endmacro()

file(GLOB TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
message(STATUS "Test headers: ${TEST_HEADERS}")

function(add_aligator_test2 test_name dependencies)
  set(test_file ${test_name}.cpp)
  set(test_name "${PROJECT_NAME}-test-cpp-${test_name}")
  message(STATUS "Adding cpp test: ${test_name} (${test_file})")

  add_unit_test(${test_name} ${test_file} ${TEST_HEADERS})
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  set_standard_output_directory(${test_name})
  target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(
    ${test_name}
    PRIVATE ${PROJECT_NAME} Catch2::Catch2WithMain ${dependencies}
  )
  # catch_discover_tests(${test_name})
endfunction()

set(
  TEST_NAMES
  arena-matrix
  continuous
  constraints
  costs
  integrators
  lqr
  problem
  manifolds
  solver-storage
  utils
)

if(BUILD_WITH_PINOCCHIO_SUPPORT)
  list(APPEND TEST_NAMES forces cycling mpc-cycle)
endif()

if(CHECK_RUNTIME_MALLOC)
  list(APPEND TEST_NAMES nomalloc)
endif()

if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  foreach(test_name ${TEST_NAMES})
    add_aligator_test2(${test_name} "")
  endforeach()
  add_subdirectory(gar)

  if(BUILD_CROCODDYL_COMPAT)
    add_subdirectory(compat)
  endif()
endif()

# Python tests
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
