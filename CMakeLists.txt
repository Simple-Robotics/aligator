#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME proxddp)
set(PROJECT_DESCRIPTION "A primal-dual augmented Lagrangian-type solver for nonlinear trajectory optimization.")
set(PROJECT_URL "https://github.com/Simple-Robotics/proxddp")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES True)

set(CXX_DISABLE_WERROR True)

set(CMAKE_CXX_EXTENSIONS OFF)
set(PROJECT_USE_CMAKE_EXPORT True)

option(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ide.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/apple.cmake)
include(CMakeDependentOption)

if(WIN32)
  set(LINK copy_if_different)
else(WIN32)
  set(LINK create_symlink)
endif(WIN32)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# --- OPTIONS ----------------------------------------
option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(BUILD_WITH_VERSION_SUFFIX "Build libraries with version appended to suffix" OFF)
option(USE_CUSTOM_LDLT "Use our custom LDLT solver" ON)
option(ENABLE_TEMPLATE_INSTANTIATION "Template instantiation of the main library" ON)

# --- OPTIONAL DEPENDENCIES -------------------------
option(BUILD_WITH_PINOCCHIO_SUPPORT "Build the library with support for Pinocchio" ON)
option(BUILD_WITH_CASADI_SUPPORT "Build the library with the support of CASADI" OFF)
option(BUILD_CROCODDYL_COMPAT "Build the Crocoddyl compatibility module" OFF)
option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" ON)
cmake_dependent_option(GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" ON BUILD_PYTHON_INTERFACE OFF)


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

macro(TAG_LIBRARY_VERSION target)
  set_target_properties(${target} PROPERTIES SOVERSION ${PROJECT_VERSION})
endmacro(TAG_LIBRARY_VERSION)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
ADD_PROJECT_DEPENDENCY(Eigen3 3.3.7 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.3.7")
ADD_PROJECT_DEPENDENCY(fmt 6.1.2 REQUIRED PKG_CONFIG_REQUIRES "fmt >= 6.1.2")

# Variable containing all the cflags definition relative to optional dependencies
set(CFLAGS_DEPENDENCIES)

if(BUILD_WITH_CASADI_SUPPORT)
  ADD_PROJECT_DEPENDENCY(casadi 3.4.5 REQUIRED PKG_CONFIG_REQUIRES "casadi >= 3.4.5")
endif(BUILD_WITH_CASADI_SUPPORT)

if(BUILD_WITH_OPENMP_SUPPORT)
  ADD_PROJECT_DEPENDENCY(OpenMP REQUIRED)
endif(BUILD_WITH_OPENMP_SUPPORT)

set(BOOST_REQUIRED_COMPONENTS filesystem serialization system)

SET_BOOST_DEFAULT_OPTIONS()
EXPORT_BOOST_DEFAULT_OPTIONS()
ADD_PROJECT_DEPENDENCY(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

if(BUILD_PYTHON_INTERFACE)
  set(PYTHON_COMPONENTS Interpreter Development.Module NumPy Development)
  ADD_PROJECT_DEPENDENCY(eigenpy 2.8.0 REQUIRED PKG_CONFIG_REQUIRES "eigenpy >= 2.8.0")
  set(PYLIB_NAME "py${PROJECT_NAME}")
  set(${PYLIB_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.python_implementation())"
    OUTPUT_VARIABLE _python_implementation_value
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  message(STATUS "Python compiler: ${_python_implementation_value}")

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import proxnlp"
  RESULT_VARIABLE _proxnlp_python_bindings_not_found
  OUTPUT_QUIET
  ERROR_QUIET)
  if(_proxnlp_python_bindings_not_found EQUAL 0)
    message(STATUS "Found proxnlp Python bindings.")
  else()
    message(FATAL_ERROR "proxnlp Python bindings NOT found.")
  endif()
  unset(_proxnlp_python_bindings_not_found)

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pinocchio"
  RESULT_VARIABLE _pinocchio_python_bindings_not_found
  OUTPUT_QUIET
  ERROR_QUIET)
  if(_pinocchio_python_bindings_not_found EQUAL 0)
    message(STATUS "Found pinocchio Python bindings.")
    set(BUILD_WITH_PINOCCHIO_PYTHON_BINDINGS True)
  else()
    message(STATUS "pinocchio Python bindings NOT found.")
  endif()
  unset(_pinocchio_python_bindings_not_found)

endif(BUILD_PYTHON_INTERFACE)

ADD_PROJECT_DEPENDENCY(proxnlp 0.2.3 REQUIRED)

set(LIB_SOURCES
    src/utils/logger.cpp)

file(GLOB_RECURSE LIB_HEADERS
     ${PROJECT_SOURCE_DIR}/include/proxddp/*.hpp
     ${PROJECT_SOURCE_DIR}/include/proxddp/*.hxx
     )

if(ENABLE_TEMPLATE_INSTANTIATION)

  file(GLOB_RECURSE LIB_TEMPLATE_SOURCES
       ${PROJECT_SOURCE_DIR}/src/core/*.cpp
       ${PROJECT_SOURCE_DIR}/src/fddp/*.cpp
       )
  file(GLOB_RECURSE LIB_TEMPLATE_DECLARATIONS
       ${PROJECT_SOURCE_DIR}/include/proxddp/*.txx
       )
  list(APPEND LIB_HEADERS ${LIB_TEMPLATE_DECLARATIONS})
  list(APPEND LIB_SOURCES ${LIB_TEMPLATE_SOURCES})

endif(ENABLE_TEMPLATE_INSTANTIATION)

list(FILTER LIB_HEADERS EXCLUDE REGEX ${PROJECT_SOURCE_DIR}/include/proxddp/compat/*)

function(create_library)
  add_library(${PROJECT_NAME} SHARED ${LIB_HEADERS} ${LIB_SOURCES})
  set_target_properties(${PROJECT_NAME}
    PROPERTIES
    LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
    VERSION ${PROJECT_VERSION})

  if(BUILD_LIBRARY_WITH_VERSION_SUFFIX)
    TAG_LIBRARY_VERSION(${PROJECT_NAME})
  endif()

  if(BUILD_WITH_PINOCCHIO_SUPPORT)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROXDDP_WITH_PINOCCHIO)
    target_link_libraries(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)
  endif(BUILD_WITH_PINOCCHIO_SUPPORT)

  if(USE_CUSTOM_LDLT)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROXDDP_CUSTOM_LDLT)
  endif(USE_CUSTOM_LDLT)

  if(BUILD_WITH_OPENMP_SUPPORT)
    add_compile_definitions(PROXDDP_MULTITHREADING)
    link_libraries(OpenMP::OpenMP_CXX)
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
    install(FILES ${headerFileRelative}
            DESTINATION ${headerPath}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
  endforeach(headerFile ${LIB_HEADERS})

endfunction()

create_library()

ADD_HEADER_GROUP(LIB_HEADERS)
ADD_SOURCE_GROUP(LIB_SOURCES)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${TARGETS_EXPORT_NAME}
        INCLUDES DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})

if(BUILD_CROCODDYL_COMPAT)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DWITH_CROCODDYL_COMPAT)
  set(CROC_COMPAT_LIB_NAME ${PROJECT_NAME}_croc_compat)
  add_subdirectory(compat/crocoddyl)
endif()

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

add_subdirectory(examples)
IF(BUILD_TESTING)
  add_subdirectory(tests)
ENDIF()

# --- PACKAGING ----------------------------------------------------------------
macro(EXPORT_VARIABLE var_name var_value)
  get_directory_property(has_parent PARENT_DIRECTORY)
  if(has_parent)
    set(${var_name} ${var_value} PARENT_SCOPE)
  else()
    set(${var_name} ${var_value})
  endif()
endmacro(EXPORT_VARIABLE var_name var_value)

if(BUILD_WITH_CASADI_SUPPORT)
  EXPORT_VARIABLE(PROXDDP_USE_CASADI ON)
  set(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PROXDDP_USE_CASADI \"\")")
endif()
if(BUILD_PYTHON_INTERFACE)
  EXPORT_VARIABLE(PROXDDP_WITH_PYTHON_INTERFACE ON)
  set(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PROXDDP_WITH_PYTHON_INTERFACE \"\")")
endif()

PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_REQUIRED_COMPONENTS})

foreach(cflags ${CFLAGS_DEPENDENCIES})
  PKG_CONFIG_APPEND_CFLAGS(${cflags})
endforeach(cflags ${CFLAGS_DEPENDENCIES})

# Install catkin package.xml
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
