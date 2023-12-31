#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#
include(${JRL_CMAKE_MODULES}/python.cmake)
if(GENERATE_PYTHON_STUBS)
  include(${JRL_CMAKE_MODULES}/stubs.cmake)
endif(GENERATE_PYTHON_STUBS)

file(GLOB PY_SOURCES src/*.cpp src/modelling/*.cpp)

set(PY_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/python)
file(GLOB PY_HEADERS ${PY_HEADER_DIR}/*.hpp ${PY_HEADER_DIR}/modelling/*.hpp
     ${PY_HEADER_DIR}/utils/*.hpp)

if(BUILD_CROCODDYL_COMPAT)
  list(APPEND PY_HEADERS ${PY_HEADER_DIR}/compat/croco.hpp)
  list(APPEND PY_SOURCES src/compat/expose-croco-compat.cpp)
endif()

function(make_bindings)
  add_library(${PYLIB_NAME} SHARED ${PY_HEADERS} ${PY_SOURCES})
  add_library(aligator::python ALIAS ${PYLIB_NAME})

  set_target_properties(
    ${PYLIB_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_NAME}
               PREFIX ""
               SUFFIX ${PYTHON_EXT_SUFFIX}
               VERSION ${PROJECT_VERSION})
  if(UNIX AND NOT APPLE)
    get_relative_rpath(${${PYLIB_NAME}_INSTALL_DIR} PYLIB_INSTALL_RPATH)
    set_target_properties(${PYLIB_NAME} PROPERTIES INSTALL_RPATH "${PYLIB_INSTALL_RPATH}")
  endif()

  # deprecated functions are exposed up until they are removed. remove warnings.
  target_compile_options(${PYLIB_NAME} PRIVATE "-Wno-deprecated-declarations")
  target_compile_definitions(${PYLIB_NAME} PRIVATE MODULE_NAME=${PYLIB_NAME})
  target_include_directories(${PYLIB_NAME}
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_link_libraries(${PYLIB_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)

  if(BUILD_CROCODDYL_COMPAT)
    target_link_libraries(${PYLIB_NAME} PUBLIC aligator::croc_compat)
  endif(BUILD_CROCODDYL_COMPAT)

  if(BUILD_WITH_OPENMP_SUPPORT)
    target_link_libraries(${PYLIB_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${PYLIB_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
  endif(BUILD_WITH_OPENMP_SUPPORT)

  install(
    TARGETS ${PYLIB_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION ${${PYLIB_NAME}_INSTALL_DIR})

endfunction()

make_bindings()

install(
  DIRECTORY ${PY_HEADER_DIR}
  DESTINATION "include/${PROJECT_NAME}"
  FILES_MATCHING
  PATTERN "*.hpp")

if(IS_ABSOLUTE ${PYTHON_SITELIB})
  set(ABSOLUTE_PYTHON_SITELIB ${PYTHON_SITELIB})
else()
  set(ABSOLUTE_PYTHON_SITELIB ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB})
endif()
# --- GENERATE STUBS
if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_NAME} ${ABSOLUTE_PYTHON_SITELIB}
                 ${PYLIB_NAME})
endif(GENERATE_PYTHON_STUBS)

set(${PYLIB_NAME}_PYFILES __init__.py)
set(${PYLIB_NAME}_PYFILES_UTILS __init__.py plotting.py)

foreach(pyfile ${${PYLIB_NAME}_PYFILES})
  python_install_on_site(${PROJECT_NAME} ${pyfile})
endforeach()

foreach(pyfile ${${PYLIB_NAME}_PYFILES_UTILS})
  python_install_on_site(${PROJECT_NAME}/utils ${pyfile})
endforeach()
