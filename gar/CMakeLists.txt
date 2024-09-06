#
# Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)

file(
  GLOB HEADERS
  RELATIVE ${CMAKE_CURRENT_LIST_DIR}/include
  CONFIGURE_DEPENDS include/aligator/gar/*.hpp include/aligator/gar/*.hxx
)

add_library(aligator_gar SHARED ${SOURCES})
target_include_directories(
  aligator_gar
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include/aligator/gar>
)
target_link_libraries(aligator_gar PUBLIC proxsuite-nlp::proxsuite-nlp)
if(BUILD_WITH_OPENMP_SUPPORT)
  target_link_libraries(aligator_gar PUBLIC OpenMP::OpenMP_CXX)
endif()
# Link to tracy, and add installed tracy headers to install interface
# for other targets to consume us without having to get tracy.
# target_link_libraries(aligator_gar PRIVATE Tracy::TracyClient)
target_link_libraries(aligator_gar PRIVATE aligator_profiling)

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
