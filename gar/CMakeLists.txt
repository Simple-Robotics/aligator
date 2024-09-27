#
# Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)

file(
  GLOB HEADERS
  RELATIVE ${CMAKE_CURRENT_LIST_DIR}/include
  CONFIGURE_DEPENDS include/aligator/gar/*.hpp include/aligator/gar/*.hxx
)

add_library(aligator_gar SHARED)
target_include_directories(aligator_gar PRIVATE ${PROJECT_SOURCE_DIR}/include/aligator)
target_link_libraries(aligator_gar PUBLIC proxsuite-nlp::proxsuite-nlp)
if(BUILD_WITH_OPENMP_SUPPORT)
  target_link_libraries(aligator_gar PUBLIC OpenMP::OpenMP_CXX)
endif()
if(ALIGATOR_TRACY_ENABLE)
  target_link_libraries(aligator_gar PUBLIC Tracy::TracyClient)
endif()

add_library(aligator::gar ALIAS aligator_gar)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  target_link_libraries(aligator_gar PUBLIC CHOLMOD::CHOLMOD)
endif()

if(ENABLE_TEMPLATE_INSTANTIATION)
  file(
    GLOB TEMPL_DECL_HEADERS
    RELATIVE ${CMAKE_CURRENT_LIST_DIR}/include
    include/aligator/gar/*.txx
  )
  list(APPEND HEADERS ${TEMPL_DECL_HEADERS})
  target_sources(aligator_gar PRIVATE ${SOURCES})
endif()

set_target_properties(aligator_gar PROPERTIES VERSION ${PROJECT_VERSION})
set_standard_output_directory(aligator_gar)

install(
  TARGETS aligator_gar
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
foreach(header ${HEADERS})
  get_filename_component(DIR ${header} DIRECTORY)
  install(FILES include/${header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${DIR})
endforeach()
