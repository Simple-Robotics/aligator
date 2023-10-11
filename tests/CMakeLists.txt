#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

macro(ADD_TEST_CFLAGS test_name flag)
  set_property(
    TARGET ${test_name}
    APPEND_STRING
    PROPERTY COMPILE_FLAGS " ${flag}")
endmacro()

function(get_cpp_test_name name dir out_var)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" prefix ${dir})
  set(${out_var}
      "test-cpp-${prefix}${name}"
      PARENT_SCOPE)
endfunction(get_cpp_test_name)

file(GLOB TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
message(STATUS "Test headers: ${TEST_HEADERS}")

function(add_aligator_test name)
  get_cpp_test_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} test_name)
  set(test_file ${name}.cpp)
  message(STATUS "Adding C++ test: ${test_file}")

  add_unit_test(${test_name} ${test_file} ${TEST_HEADERS})
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  # define macros required by boost_test
  # see: https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/usage_variants.html
  # define module name, replace '-' by '_'
  set(MODULE_NAME "${name}Test")
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})

  add_test_cflags(${test_name} "-DBOOST_TEST_DYN_LINK")
  add_test_cflags(${test_name} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  target_link_libraries(${test_name} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${test_name} PRIVATE Boost::unit_test_framework)
endfunction(add_aligator_test)

set(TEST_NAMES
    integrators
    problem
    costs
    continuous
    utils
    solver-storage
    parallel-lqr
    gar-riccati
    schur3x3)

foreach(test_name ${TEST_NAMES})
  add_aligator_test(${test_name})
endforeach(test_name)

# PYTHON TESTS
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

if(BUILD_CROCODDYL_COMPAT)
  add_subdirectory(compat)
endif()

target_link_libraries(test-cpp-schur3x3 PRIVATE OpenMP::OpenMP_CXX)
