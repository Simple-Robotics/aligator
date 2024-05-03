#
# Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)

file(GLOB HEADERS CONFIGURE_DEPENDS include/aligator/gar/*.hpp)

add_library(aligator_gar SHARED ${HEADERS})
target_include_directories(aligator_gar PRIVATE ${PROJECT_SOURCE_DIR}/include/aligator)
target_include_directories(
  aligator_gar PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include/aligator/gar>
)
target_link_libraries(aligator_gar PUBLIC proxsuite-nlp::proxsuite-nlp)
if(BUILD_WITH_OPENMP_SUPPORT)
  target_link_libraries(aligator_gar PUBLIC OpenMP::OpenMP_CXX)
endif(BUILD_WITH_OPENMP_SUPPORT)
target_link_libraries(aligator_gar PUBLIC Tracy::TracyClient)

add_library(aligator::gar ALIAS aligator_gar)

if(ENABLE_TEMPLATE_INSTANTIATION)
  file(GLOB TEMPL_DECL_HEADERS include/aligator/gar/*.txx)
  list(APPEND HEADERS ${TEMPL_DECL_HEADERS})
  target_sources(aligator_gar PRIVATE ${SOURCES} ${TEMPL_DECL_HEADERS})
endif(ENABLE_TEMPLATE_INSTANTIATION)

set_target_properties(aligator_gar PROPERTIES VERSION ${PROJECT_VERSION} PUBLIC_HEADER "${HEADERS}")
set_standard_output_directory(aligator_gar)

install(
  TARGETS aligator_gar
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aligator/gar
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
