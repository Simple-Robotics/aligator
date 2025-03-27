#
# Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#
include(${JRL_CMAKE_MODULES}/python.cmake)
if(GENERATE_PYTHON_STUBS)
  include(${JRL_CMAKE_MODULES}/stubs.cmake)
endif()

file(
  GLOB PY_SOURCES
  CONFIGURE_DEPENDS
  src/*.cpp
  src/gar/*.cpp
  src/modelling/*.cpp
)

set(PY_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/python)
file(
  GLOB PY_HEADERS
  CONFIGURE_DEPENDS
  ${PY_HEADER_DIR}/*.hpp
  ${PY_HEADER_DIR}/modelling/*.hpp
  ${PY_HEADER_DIR}/utils/*.hpp
  ${PY_HEADER_DIR}/gar/*.hpp
)

if(BUILD_CROCODDYL_COMPAT)
  list(APPEND PY_SOURCES src/compat/expose-croco-compat.cpp)
endif()

function(make_bindings)
  Python3_add_library(
    ${PYLIB_NAME}
    MODULE
    WITH_SOABI
    ${PY_HEADERS}
    ${PY_SOURCES}
  )
  add_library(aligator::python ALIAS ${PYLIB_NAME})

  set_target_properties(
    ${PYLIB_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_NAME}
  )
  if(UNIX)
    GET_RELATIVE_RPATH(${${PYLIB_NAME}_INSTALL_DIR} PYLIB_INSTALL_RPATH)
    set_target_properties(
      ${PYLIB_NAME}
      PROPERTIES INSTALL_RPATH "${PYLIB_INSTALL_RPATH}"
    )
  endif()

  # deprecated functions are exposed up until they are removed. remove warnings.
  target_compile_options(${PYLIB_NAME} PRIVATE "-Wno-deprecated-declarations")
  target_compile_definitions(${PYLIB_NAME} PRIVATE MODULE_NAME=${PYLIB_NAME})
  target_include_directories(
    ${PYLIB_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

  target_link_libraries(${PYLIB_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
  target_include_directories(
    ${PYLIB_NAME}
    PUBLIC $<TARGET_PROPERTY:proxsuite-nlp::pyproxsuite_nlp,INCLUDE_DIRECTORIES>
  )

  if(BUILD_CROCODDYL_COMPAT)
    target_link_libraries(${PYLIB_NAME} PUBLIC aligator::croc_compat)
  endif()

  install(
    TARGETS ${PYLIB_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aligator/python
    LIBRARY DESTINATION ${${PYLIB_NAME}_INSTALL_DIR}
  )
endfunction()

make_bindings()

install(
  DIRECTORY ${PY_HEADER_DIR}
  DESTINATION "include/${PROJECT_NAME}"
  FILES_MATCHING
  PATTERN "*.hpp"
)

if(IS_ABSOLUTE ${PYTHON_SITELIB})
  set(ABSOLUTE_PYTHON_SITELIB ${PYTHON_SITELIB})
else()
  set(ABSOLUTE_PYTHON_SITELIB ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB})
endif()
# --- GENERATE STUBS
if(GENERATE_PYTHON_STUBS)
  PYTHON_BUILD_GET_TARGET(python_build_target_name)
  LOAD_STUBGEN()
  # Set PYWRAP and PROJECT_NAME as stubs dependencies.
  # PROJECT_NAME is mandatory (even if it's a PYWRAP dependency)
  # to find PROJECT_NAME name DLL on windows.
  GENERATE_STUBS(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_NAME}
    ${ABSOLUTE_PYTHON_SITELIB}
    ${PYLIB_NAME}
    ${python_build_target_name}
  )
endif()

set(${PYLIB_NAME}_PYFILES __init__.py)
set(${PYLIB_NAME}_PYFILES_UTILS __init__.py plotting.py)

foreach(pyfile ${${PYLIB_NAME}_PYFILES})
  PYTHON_INSTALL_ON_SITE(${PROJECT_NAME} ${pyfile})
endforeach()

foreach(pyfile ${${PYLIB_NAME}_PYFILES_UTILS})
  PYTHON_INSTALL_ON_SITE(${PROJECT_NAME}/utils ${pyfile})
endforeach()
