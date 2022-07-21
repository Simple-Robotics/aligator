include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)

file(GLOB_RECURSE ${PYLIB_NAME}_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB_RECURSE ${PYLIB_NAME}_HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/**.hpp)

set(PY_HEADERS ${${PYLIB_NAME}_HEADERS})
set(PY_SOURCES ${${PYLIB_NAME}_SOURCES})

function(make_bindings)
  add_library(${PYLIB_NAME} SHARED ${PY_HEADERS} ${PY_SOURCES})

  target_include_directories(${PYLIB_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
  set_target_properties(${PYLIB_NAME}
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_NAME}
    PREFIX ""
    SUFFIX ${PYTHON_EXT_SUFFIX}
    VERSION ${PROJECT_VERSION}
    )
  if(UNIX AND NOT APPLE)
    set_target_properties(${PYLIB_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../../..")
  endif()

  target_link_libraries(${PYLIB_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
  target_link_libraries(${PYLIB_NAME} PRIVATE fmt::fmt)
  target_link_boost_python(${PYLIB_NAME} PUBLIC)
  install(TARGETS ${PYLIB_NAME}
          EXPORT ${TARGETS_EXPORT_NAME}
          DESTINATION ${${PYLIB_NAME}_INSTALL_DIR})

endfunction()

make_bindings()

# symlink the headers here

file(MAKE_DIRECTORY
     ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/python
     ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/python/compat
     ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/python/modelling
     ${CMAKE_CURRENT_BINARY_DIR}/examples)

foreach(pyheader ${PY_HEADERS})
  string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" pyheader ${pyheader})
  get_filename_component(headerPath ${pyheader} PATH)
  execute_process(COMMAND ${CMAKE_COMMAND} -E ${LINK}
                  ${CMAKE_CURRENT_SOURCE_DIR}/${pyheader}
                  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/python/${pyheader})
  install(FILES ${pyheader}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/python/${headerPath}
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
endforeach(pyheader)

if(IS_ABSOLUTE ${PYTHON_SITELIB})
  set(ABSOLUTE_PYTHON_SITELIB ${PYTHON_SITELIB})
else()
  set(ABSOLUTE_PYTHON_SITELIB ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB})
endif()
# --- GENERATE STUBS
if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_NAME} ${ABSOLUTE_PYTHON_SITELIB})
endif(GENERATE_PYTHON_STUBS)

set(${PYLIB_NAME}_PYFILES
    __init__.py
    utils.py)

foreach(pyfile ${${PYLIB_NAME}_PYFILES})
  python_install_on_site(${PROJECT_NAME} ${pyfile})
endforeach()
