#
# Copyright (C) 2023 LAAS-CNRS, INRIA
#
# GAR (General Augmented lagrangian Riccati) is a set of algorithms for the
# fast solving of proximal and constrained LQR-type linear systems.

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/riccati.cpp)

file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/aligator/gar/*.hpp)

add_library(aligator_gar SHARED ${HEADERS})
target_include_directories(aligator_gar PRIVATE ${PROJECT_SOURCE_DIR}/include/aligator)
target_link_libraries(aligator_gar PUBLIC proxnlp::proxnlp)
add_library(aligator::gar ALIAS aligator_gar)

if(ENABLE_TEMPLATE_INSTANTIATION)
  target_sources(aligator_gar PRIVATE ${SOURCES})
endif(ENABLE_TEMPLATE_INSTANTIATION)

install(
  TARGETS aligator_gar
  EXPORT ${TARGETS_EXPORT_NAME}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})
