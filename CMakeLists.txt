#
# Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#

cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME aligator)
set(PROJECT_DESCRIPTION
    "A primal-dual augmented Lagrangian-type solver for nonlinear trajectory optimization."
)
set(PROJECT_URL "https://github.com/Simple-Robotics/aligator")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES True)

set(CXX_DISABLE_WERROR True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(PROJECT_USE_CMAKE_EXPORT True)

option(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

set(DOXYGEN_USE_MATHJAX NO)
set(DOXYGEN_USE_TEMPLATE_CSS YES)
set(DOXYGEN_HTML_HEADER "${CMAKE_SOURCE_DIR}/doc/header.html")
set(DOXYGEN_HTML_STYLESHEET "")

include(FetchContent)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    )
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    FetchContent_Declare(
      "jrl-cmakemodules" GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git"
    )
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

include(${JRL_CMAKE_MODULES}/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE OFF)

include(${JRL_CMAKE_MODULES}/boost.cmake)
include(${JRL_CMAKE_MODULES}/ide.cmake)
include(${JRL_CMAKE_MODULES}/apple.cmake)
if(APPLE) # Use the handmade approach
  set(CMAKE_MODULE_PATH ${JRL_CMAKE_MODULES}/find-external/OpenMP ${CMAKE_MODULE_PATH})
elseif(UNIX)
  if(${CMAKE_VERSION} VERSION_GREATER "3.20.0" OR ${CMAKE_VERSION} VERSION_EQUAL "3.20.0")
    set(CMAKE_MODULE_PATH ${JRL_CMAKE_MODULES}/find-external/OpenMP ${CMAKE_MODULE_PATH})
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
      CACHE STRING "Choose the type of build." FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

function(set_standard_output_directory target)
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
               ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  )
endfunction()

# --- OPTIONS ----------------------------------------
option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(BUILD_WITH_VERSION_SUFFIX "Build libraries with version appended to suffix" OFF)
option(ENABLE_TEMPLATE_INSTANTIATION "Template instantiation of the main library" ON)

# --- OPTIONAL DEPENDENCIES -------------------------
option(BUILD_WITH_PINOCCHIO_SUPPORT "Build the library with support for Pinocchio" ON)
option(BUILD_CROCODDYL_COMPAT "Build the Crocoddyl compatibility module" OFF)
option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" ON)
option(BUILD_WITH_CHOLMOD_SUPPORT "Build with support for backends enabled by Cholmod" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)
cmake_dependent_option(
  GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" ON
  BUILD_PYTHON_INTERFACE OFF
)

option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)

# Variable containing all the cflags definition relative to optional dependencies
# and options
set(CFLAGS_DEPENDENCIES)

if(INITIALIZE_WITH_NAN)
  message(STATUS "Initialize with NaN all the Eigen entries.")
  add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
endif(INITIALIZE_WITH_NAN)

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_compile_definitions(ALIGATOR_EIGEN_CHECK_MALLOC)
  add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)

if(ENABLE_TEMPLATE_INSTANTIATION)
  add_compile_definitions(ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION)
  list(APPEND CFLAGS_DEPENDENCIES "-DALIGATOR_ENABLE_TEMPLATE_INSTANTIATION")
endif(ENABLE_TEMPLATE_INSTANTIATION)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
add_project_dependency(Eigen3 3.3.7 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.3.7")
add_project_dependency(fmt "9.1.0...<11" REQUIRED PKG_CONFIG_REQUIRES "fmt >= 9.1.0")

if(BUILD_WITH_OPENMP_SUPPORT)
  add_project_dependency(OpenMP REQUIRED)
  add_compile_definitions(ALIGATOR_MULTITHREADING)
endif(BUILD_WITH_OPENMP_SUPPORT)

if(BUILD_WITH_PINOCCHIO_SUPPORT)
  message(STATUS "Building with Pinocchio support.")
  add_compile_definitions(ALIGATOR_WITH_PINOCCHIO)
  list(APPEND CFLAGS_DEPENDENCIES "-DALIGATOR_WITH_PINOCCHIO")
endif()

if(BUILD_CROCODDYL_COMPAT)
  message(STATUS "Building with Crocoddyl compatibility.")
  add_compile_definitions(ALIGATOR_WITH_CROCODDYL_COMPAT)
  list(APPEND CFLAGS_DEPENDENCIES "-DALIGATOR_WITH_CROCODDYL_COMPAT")
endif()

if(BUILD_WITH_CHOLMOD_SUPPORT)
  set(CMAKE_MODULE_PATH ${JRL_CMAKE_MODULES}/find-external/CHOLMOD ${CMAKE_MODULE_PATH})
  add_project_dependency(CHOLMOD REQUIRED)
  message(
    STATUS "Build with CHOLMOD support (LGPL). See CHOLMOD/Doc/License.txt for further details."
  )
  add_compile_definitions(ALIGATOR_WITH_CHOLMOD)
endif()

set(BOOST_REQUIRED_COMPONENTS filesystem)
set_boost_default_options()
export_boost_default_options()
add_project_dependency(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

if(BUILD_PYTHON_INTERFACE)
  set(PYTHON_COMPONENTS Interpreter Development.Module NumPy Development)
  add_project_dependency(eigenpy 3.4.0 REQUIRED PKG_CONFIG_REQUIRES "eigenpy >= 3.4.0")
  set(PYLIB_NAME "py${PROJECT_NAME}")
  set(${PYLIB_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.python_implementation())"
    OUTPUT_VARIABLE _python_implementation_value
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  )
  message(STATUS "Python compiler: ${_python_implementation_value}")

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import proxsuite_nlp"
    RESULT_VARIABLE _proxsuite_nlp_python_bindings_not_found
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_proxsuite_nlp_python_bindings_not_found EQUAL 0)
    message(STATUS "Found proxsuite_nlp Python bindings.")
  else()
    message(FATAL_ERROR "proxsuite_nlp Python bindings NOT found.")
  endif()
  unset(_proxsuite_nlp_python_bindings_not_found)

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pinocchio"
    RESULT_VARIABLE _pinocchio_python_bindings_not_found
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_pinocchio_python_bindings_not_found EQUAL 0)
    message(STATUS "Found pinocchio Python bindings.")
    set(BUILD_WITH_PINOCCHIO_PYTHON_BINDINGS True)
  else()
    message(STATUS "pinocchio Python bindings NOT found.")
  endif()
  unset(_pinocchio_python_bindings_not_found)

endif(BUILD_PYTHON_INTERFACE)

add_project_dependency(proxsuite-nlp 0.6.1 REQUIRED PKG_CONFIG_REQUIRES "proxsuite-nlp >= 0.6.1")

set(LIB_SOURCES src/utils/logger.cpp)

file(GLOB_RECURSE LIB_HEADERS ${PROJECT_SOURCE_DIR}/include/aligator/*.hpp
     ${PROJECT_SOURCE_DIR}/include/aligator/*.hxx
)

if(ENABLE_TEMPLATE_INSTANTIATION)

  file(GLOB_RECURSE LIB_TEMPLATE_SOURCES ${PROJECT_SOURCE_DIR}/src/core/*.cpp
       ${PROJECT_SOURCE_DIR}/src/solvers/fddp/*.cpp ${PROJECT_SOURCE_DIR}/src/solvers/proxddp/*.cpp
       ${PROJECT_SOURCE_DIR}/src/modelling/*.cpp
  )
  file(GLOB_RECURSE LIB_TEMPLATE_DECLARATIONS ${PROJECT_SOURCE_DIR}/include/aligator/*.txx)
  list(APPEND LIB_HEADERS ${LIB_TEMPLATE_DECLARATIONS})
  list(APPEND LIB_SOURCES ${LIB_TEMPLATE_SOURCES})

endif(ENABLE_TEMPLATE_INSTANTIATION)

list(FILTER LIB_HEADERS EXCLUDE REGEX ${PROJECT_SOURCE_DIR}/include/aligator/compat/*)

option(ENABLE_TRACY_PROFILING "Enable Tracy profiler" OFF)
option(DOWNLOAD_TRACY "Use FetchContent to install Tracy." ON)
if(DOWNLOAD_TRACY)
  FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/Simple-Robotics/tracy.git
    GIT_TAG patches
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
  FetchContent_GetProperties(tracy)
  if(NOT tracy_POPULATED)
    FetchContent_Populate(tracy)
    set(TRACY_STATIC
        ON
        CACHE INTERNAL ""
    )
    set(TRACY_ENABLE
        ${ENABLE_TRACY_PROFILING}
        CACHE INTERNAL ""
    )
    add_subdirectory(${tracy_SOURCE_DIR} ${tracy_BINARY_DIR} EXCLUDE_FROM_ALL)
    # Extract the target include directories, set as system
    get_target_property(tracy_INCLUDE_DIR TracyClient INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(
      TracyClient PROPERTIES POSITION_INDEPENDENT_CODE True INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                                            "${tracy_INCLUDE_DIR}"
    )
  endif()
else()
  # assume it is installed somewhere
  find_package(Tracy REQUIRED)
  set_target_properties(Tracy::TracyClient PROPERTIES POSITION_INDEPENDENT_CODE True)
endif()

add_subdirectory(gar)

# Create the main shared library.
function(create_library)
  add_library(${PROJECT_NAME} SHARED ${LIB_HEADERS} ${LIB_SOURCES})
  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES LINKER_LANGUAGE CXX
               VERSION ${PROJECT_VERSION}
               INSTALL_RPATH "\$ORIGIN"
  )
  set_standard_output_directory(${PROJECT_NAME})

  target_link_libraries(${PROJECT_NAME} PUBLIC aligator::gar)
  # Extract the compile definitions of the project for export
  get_directory_property(CURRENT_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${CURRENT_COMPILE_DEFINITIONS})

  if(BUILD_WITH_PINOCCHIO_SUPPORT)
    target_link_libraries(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)
  endif(BUILD_WITH_PINOCCHIO_SUPPORT)

  if(BUILD_WITH_OPENMP_SUPPORT)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
  endif(BUILD_WITH_OPENMP_SUPPORT)

  target_link_libraries(${PROJECT_NAME} PUBLIC proxsuite-nlp::proxsuite-nlp)
  target_link_libraries(${PROJECT_NAME} PUBLIC Boost::boost)
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
      PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE
    )
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
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
)

if(DOWNLOAD_TRACY)
  install(
    TARGETS TracyClient
    EXPORT ${TARGETS_EXPORT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/aligator
    ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/aligator
  )
  message(STATUS "Installing TracyClient target to ${CMAKE_INSTALL_FULL_LIBDIR}/aligator")
endif()

if(BUILD_CROCODDYL_COMPAT)
  add_project_dependency(crocoddyl REQUIRED)
  add_subdirectory(src/compat/crocoddyl)
endif()

add_subdirectory(bindings)

# benchmarks, examples, and tests

macro(create_ex_or_bench exfile exname)
  add_executable(${exname} ${exfile})
  message(STATUS "Adding cpp example ${exname}")
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  set_standard_output_directory(${exname})
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${exname} PUBLIC example-robot-data::example-robot-data)
endmacro()

macro(target_add_example_robot_data target_name)
  target_link_libraries(${target_name} PRIVATE example-robot-data::example-robot-data)
endmacro()

if(BUILD_WITH_PINOCCHIO_SUPPORT AND (BUILD_EXAMPLES OR BUILD_BENCHMARKS))
  find_package(example-robot-data 4.0.9 REQUIRED)
endif()

# create an utility library to avoid recompiling crocoddyl talos arm problem
# used in examples and benchmarks
if(BUILD_CROCODDYL_COMPAT AND (BUILD_EXAMPLES OR BUILD_BENCHMARKS))
  add_library(
    croc_talos_arm_utils STATIC ${CMAKE_SOURCE_DIR}/examples/croc-talos-arm.cpp
                                ${CMAKE_SOURCE_DIR}/examples/croc-talos-arm.hpp
  )
  target_include_directories(croc_talos_arm_utils PUBLIC ${CMAKE_SOURCE_DIR}/examples)
  target_link_libraries(
    croc_talos_arm_utils PUBLIC ${PROJECT_NAME} Boost::boost crocoddyl::crocoddyl
  )
  set_standard_output_directory(croc_talos_arm_utils)
  target_add_example_robot_data(croc_talos_arm_utils)
endif()

if(BUILD_BENCHMARKS OR BUILD_TESTING)
  add_library(gar_test_utils STATIC ${CMAKE_SOURCE_DIR}/tests/gar/test_util.cpp)
  target_link_libraries(gar_test_utils PRIVATE aligator::gar)
  set_standard_output_directory(gar_test_utils)
endif()

if(${pinocchio_VERSION} VERSION_GREATER "2.9.0")
  set(PINOCCHIO_V3 True)
endif()

if(BUILD_EXAMPLES AND PINOCCHIO_V3)
  add_library(
    talos_walk_utils STATIC ${CMAKE_SOURCE_DIR}/examples/talos-walk-utils.cpp
                            ${CMAKE_SOURCE_DIR}/examples/talos-walk-utils.hpp
  )
  target_include_directories(talos_walk_utils PUBLIC ${CMAKE_SOURCE_DIR}/examples)
  set_standard_output_directory(talos_walk_utils)
  target_link_libraries(talos_walk_utils PUBLIC ${PROJECT_NAME})
  target_add_example_robot_data(talos_walk_utils)
  function(target_add_talos_walk target)
    target_link_libraries(${target} PRIVATE talos_walk_utils)
  endfunction()
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
  find_package(benchmark REQUIRED)
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
        PARENT_SCOPE
    )
  else()
    set(${var_name} ${var_value})
  endif()
  string(APPEND PACKAGE_EXTRA_MACROS "\nset(${var_name} ${var_value})")
endmacro()

if(BUILD_WITH_PINOCCHIO_SUPPORT)
  export_variable(ALIGATOR_WITH_PINOCCHIO_SUPPORT ON)
endif()
if(ENABLE_TEMPLATE_INSTANTIATION)
  export_variable(ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION ON)
endif()
if(BUILD_PYTHON_INTERFACE)
  export_variable(ALIGATOR_WITH_PYTHON_INTERFACE ON)
endif()
if(BUILD_WITH_OPENMP_SUPPORT)
  export_variable(ALIGATOR_WITH_OPENMP_SUPPORT ON)
endif()
if(BUILD_CROCODDYL_COMPAT)
  export_variable(ALIGATOR_WITH_CROCODDYL_COMPAT ON)
  pkg_config_append_libs(aligator_croc_compat)
endif()

pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_boost_libs(${BOOST_REQUIRED_COMPONENTS})
pkg_config_append_cflags("${CFLAGS_DEPENDENCIES}")

# Install catkin package.xml
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
