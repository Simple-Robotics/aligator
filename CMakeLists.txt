#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME proxddp)
set(PROJECT_DESCRIPTION
    "A primal-dual augmented Lagrangian-type solver for nonlinear trajectory optimization.")
set(PROJECT_URL "https://github.com/Simple-Robotics/proxddp")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES True)

set(CXX_DISABLE_WERROR True)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(PROJECT_USE_CMAKE_EXPORT True)

option(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ide.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/apple.cmake)
if(APPLE) # Use the handmade approach
  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP ${CMAKE_MODULE_PATH})
elseif(UNIX)
  if(${CMAKE_VERSION} VERSION_GREATER "3.20.0" OR ${CMAKE_VERSION} VERSION_EQUAL "3.20.0")
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP ${CMAKE_MODULE_PATH})
  endif()
endif(APPLE)
include(CMakeDependentOption)

apply_default_apple_configuration()

if(WIN32)
  set(LINK copy_if_different)
else(WIN32)
  set(LINK create_symlink)
endif(WIN32)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# --- OPTIONS ----------------------------------------
option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(BUILD_WITH_VERSION_SUFFIX "Build libraries with version appended to suffix" OFF)
option(ENABLE_TEMPLATE_INSTANTIATION "Template instantiation of the main library" ON)

# --- OPTIONAL DEPENDENCIES -------------------------
option(BUILD_WITH_PINOCCHIO_SUPPORT "Build the library with support for Pinocchio" ON)
option(BUILD_CROCODDYL_COMPAT "Build the Crocoddyl compatibility module" OFF)
option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)
cmake_dependent_option(
  GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" ON
  BUILD_PYTHON_INTERFACE OFF)

option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)

if(INITIALIZE_WITH_NAN)
  message(STATUS "Initialize with NaN all the Eigen entries.")
  add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
endif(INITIALIZE_WITH_NAN)

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_definitions(-DPROXDDP_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)

if(ENABLE_TEMPLATE_INSTANTIATION)
  add_definitions(-DPROXDDP_ENABLE_TEMPLATE_INSTANTIATION)
endif(ENABLE_TEMPLATE_INSTANTIATION)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
add_project_dependency(Eigen3 3.3.7 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.3.7")
add_project_dependency(fmt "9.1.0...<11" REQUIRED PKG_CONFIG_REQUIRES "fmt >= 9.1.0")

# Variable containing all the cflags definition relative to optional dependencies
set(CFLAGS_DEPENDENCIES)

if(BUILD_WITH_OPENMP_SUPPORT)
  find_package(OpenMP REQUIRED)
endif(BUILD_WITH_OPENMP_SUPPORT)

set(BOOST_REQUIRED_COMPONENTS filesystem serialization system)

set_boost_default_options()
export_boost_default_options()
add_project_dependency(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

if(BUILD_PYTHON_INTERFACE)
  set(PYTHON_COMPONENTS Interpreter Development.Module NumPy Development)
  add_project_dependency(eigenpy 3.1.0 REQUIRED PKG_CONFIG_REQUIRES "eigenpy >= 3.1.0")
  set(PYLIB_NAME "py${PROJECT_NAME}")
  set(${PYLIB_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.python_implementation())"
    OUTPUT_VARIABLE _python_implementation_value
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  message(STATUS "Python compiler: ${_python_implementation_value}")

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import proxnlp"
    RESULT_VARIABLE _proxnlp_python_bindings_not_found
    OUTPUT_QUIET ERROR_QUIET)
  if(_proxnlp_python_bindings_not_found EQUAL 0)
    message(STATUS "Found proxnlp Python bindings.")
  else()
    message(FATAL_ERROR "proxnlp Python bindings NOT found.")
  endif()
  unset(_proxnlp_python_bindings_not_found)

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pinocchio"
    RESULT_VARIABLE _pinocchio_python_bindings_not_found
    OUTPUT_QUIET ERROR_QUIET)
  if(_pinocchio_python_bindings_not_found EQUAL 0)
    message(STATUS "Found pinocchio Python bindings.")
    set(BUILD_WITH_PINOCCHIO_PYTHON_BINDINGS True)
  else()
    message(STATUS "pinocchio Python bindings NOT found.")
  endif()
  unset(_pinocchio_python_bindings_not_found)

endif(BUILD_PYTHON_INTERFACE)

add_project_dependency(proxnlp 0.2.3 REQUIRED)

set(LIB_SOURCES src/utils/logger.cpp)

file(GLOB_RECURSE LIB_HEADERS ${PROJECT_SOURCE_DIR}/include/proxddp/*.hpp
     ${PROJECT_SOURCE_DIR}/include/proxddp/*.hxx)

if(ENABLE_TEMPLATE_INSTANTIATION)

  file(
    GLOB_RECURSE
    LIB_TEMPLATE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/core/*.cpp
    ${PROJECT_SOURCE_DIR}/src/solvers/fddp/*.cpp
    ${PROJECT_SOURCE_DIR}/src/solvers/proxddp/*.cpp
    ${PROJECT_SOURCE_DIR}/src/modelling/*.cpp
    ${PROJECT_SOURCE_DIR}/src/parlqr/*.cpp)
  file(GLOB_RECURSE LIB_TEMPLATE_DECLARATIONS ${PROJECT_SOURCE_DIR}/include/proxddp/*.txx)
  list(APPEND LIB_HEADERS ${LIB_TEMPLATE_DECLARATIONS})
  list(APPEND LIB_SOURCES ${LIB_TEMPLATE_SOURCES})

endif(ENABLE_TEMPLATE_INSTANTIATION)

list(FILTER LIB_HEADERS EXCLUDE REGEX ${PROJECT_SOURCE_DIR}/include/proxddp/compat/*)

# Create the main shared library.
function(create_library)
  add_library(${PROJECT_NAME} SHARED ${LIB_HEADERS} ${LIB_SOURCES})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX VERSION ${PROJECT_VERSION})

  if(BUILD_WITH_PINOCCHIO_SUPPORT)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROXDDP_WITH_PINOCCHIO)
    target_link_libraries(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)
  endif(BUILD_WITH_PINOCCHIO_SUPPORT)

  if(BUILD_WITH_OPENMP_SUPPORT)
    add_definitions(-DPROXDDP_MULTITHREADING)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
  endif(BUILD_WITH_OPENMP_SUPPORT)

  target_link_libraries(${PROJECT_NAME} PUBLIC proxnlp::proxnlp)
  target_link_libraries(${PROJECT_NAME} PUBLIC Boost::filesystem Boost::system Boost::serialization)
  target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)
  # set the install-tree include dirs
  # used by dependent projects to consume this target
  target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)

  foreach(headerFile ${LIB_HEADERS})
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/" "" headerFileRelative ${headerFile})
    get_filename_component(headerPath ${headerFileRelative} PATH)
    install(
      FILES ${headerFileRelative}
      DESTINATION ${headerPath}
      PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
  endforeach(headerFile ${LIB_HEADERS})

endfunction()

create_library()

add_header_group(LIB_HEADERS)
add_source_group(LIB_SOURCES)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})

if(BUILD_CROCODDYL_COMPAT)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROXDDP_WITH_CROCODDYL_COMPAT)
  add_subdirectory(src/compat/crocoddyl)
endif()

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

# benchmarks, examples, and tests

macro(create_ex_or_bench exfile exname)

  add_unit_test(${exname} ${exfile})
  message(STATUS "Adding cpp example ${exname}")
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${exname} PUBLIC example-robot-data::example-robot-data)
  if(BUILD_WITH_OPENMP_SUPPORT)
    target_link_libraries(${exname} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endmacro()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# --- PACKAGING ----------------------------------------------------------------
macro(EXPORT_VARIABLE var_name var_value)
  get_directory_property(has_parent PARENT_DIRECTORY)
  if(has_parent)
    set(${var_name}
        ${var_value}
        PARENT_SCOPE)
  else()
    set(${var_name} ${var_value})
  endif()
endmacro(
  EXPORT_VARIABLE
  var_name
  var_value)

if(BUILD_WITH_CASADI_SUPPORT)
  export_variable(PROXDDP_USE_CASADI ON)
  set(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PROXDDP_USE_CASADI \"\")")
endif()
if(BUILD_PYTHON_INTERFACE)
  export_variable(PROXDDP_WITH_PYTHON_INTERFACE ON)
  set(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PROXDDP_WITH_PYTHON_INTERFACE \"\")")
endif()

pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_boost_libs(${BOOST_REQUIRED_COMPONENTS})

foreach(cflags ${CFLAGS_DEPENDENCIES})
  pkg_config_append_cflags(${cflags})
endforeach(cflags ${CFLAGS_DEPENDENCIES})

# Install catkin package.xml
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
