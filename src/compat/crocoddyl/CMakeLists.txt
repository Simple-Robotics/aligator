set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/compat/crocoddyl)
set(
  CROC_INSTALL_INCLUDE_DIR
  ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/compat/crocoddyl
)

file(GLOB_RECURSE HEADERS ${INCLUDE_DIR}/*.hpp ${INCLUDE_DIR}/*.hxx)
add_library(aligator_croc_compat SHARED ${HEADERS})
add_library(aligator::croc_compat ALIAS aligator_croc_compat)

if(ENABLE_TEMPLATE_INSTANTIATION)
  file(GLOB SOURCES_TEMPL ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  list(APPEND HEADERS ${HEADERS_TEMPL})

  file(GLOB HEADERS_TEMPL ${INCLUDE_DIR}/*.txx)

  target_sources(aligator_croc_compat PRIVATE ${HEADERS_TEMPL} ${SOURCES_TEMPL})
endif()

set_target_properties(
  aligator_croc_compat
  PROPERTIES
    LINKER_LANGUAGE CXX
    PUBLIC_HEADER "${HEADERS}"
    INSTALL_RPATH "\$ORIGIN"
    VERSION ${PROJECT_VERSION}
)
set_standard_output_directory(aligator_croc_compat)

target_link_libraries(
  aligator_croc_compat
  PUBLIC ${PROJECT_NAME} crocoddyl::crocoddyl
)
target_compile_definitions(
  aligator_croc_compat
  PUBLIC ALIGATOR_WITH_CROCODDYL_COMPAT
)

install(
  TARGETS aligator_croc_compat
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CROC_INSTALL_INCLUDE_DIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
