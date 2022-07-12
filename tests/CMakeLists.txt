#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

ADD_PROJECT_DEPENDENCY(benchmark REQUIRED)


macro(ADD_TEST_CFLAGS test_name flag)
  set_property(TARGET ${test_name} APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
endmacro()


function(get_cpp_test_name name dir out_var)
  string(REPLACE "${TEST_DIR}" "" prefix ${dir})
  set(${out_var} "test-cpp-${prefix}${name}" PARENT_SCOPE)
endfunction(get_cpp_test_name)

set(TEST_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/generate-problem.hpp)

function(add_proxddp_test name)
  get_cpp_test_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} test_name)
  set(test_file ${name}.cpp)

  ADD_UNIT_TEST(${test_name} ${test_file} ${TEST_HEADERS})
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


  # define macros required by boost_test
  # see: https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/usage_variants.html
  # define module name, replace '-' by '_'
  set(MODULE_NAME "${name}Test")
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})

  ADD_TEST_CFLAGS(${test_name} "-DBOOST_TEST_DYN_LINK")
  ADD_TEST_CFLAGS(${test_name} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  target_link_libraries(${test_name} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${test_name} PRIVATE Boost::unit_test_framework fmt::fmt benchmark::benchmark)
endfunction(add_proxddp_test)

set(TEST_NAMES
    integrators
    problem
    continuous
    utils
    solver-storage)

foreach(test_name ${TEST_NAMES})
  add_proxddp_test(${test_name})
endforeach(test_name)

# PYTHON TESTS
if (BUILD_PYTHON_INTERFACE)
  file(GLOB PYTHON_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} python/test_*.py)
  message(STATUS "Python tests: ${PYTHON_TESTS}")
  make_directory(${CMAKE_CURRENT_BINARY_DIR}/python)

  foreach(pyfile ${PYTHON_TESTS})
    get_filename_component(pyfile_name ${pyfile} NAME)
    message(STATUS "adding python test ${pyfile_name}")
    ADD_PYTHON_UNIT_TEST(${pyfile_name} "tests/${pyfile}")
  endforeach()
endif()

if(BUILD_CROCODDYL_COMPAT)
  add_subdirectory(compat)
endif()
