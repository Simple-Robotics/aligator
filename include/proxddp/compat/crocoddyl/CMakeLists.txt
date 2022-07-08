
file(GLOB_RECURSE HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

set(SOURCES
    instantiate.cpp)

set(LIB_HEADERS ${LIB_HEADERS} ${HEADERS})

message(STATUS "Compat module: headers ${HEADERS}")

find_package(crocoddyl REQUIRED)

set(COMPAT_LIB ${PROJECT_NAME}_croc_compat)
add_library(${COMPAT_LIB} SHARED ${HEADERS} ${SOURCES})
set_target_properties(${COMPAT_LIB}
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
  VERSION ${PROJECT_VERSION})

target_link_libraries(${COMPAT_LIB} PUBLIC ${PROJECT_NAME} crocoddyl::crocoddyl)
