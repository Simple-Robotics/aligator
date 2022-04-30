#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

file(GLOB EXAMPLES_CPP
     *.cpp)


foreach(exfile ${EXAMPLES_CPP})
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "example-${exname}")
  get_filename_component(exfile ${exfile} NAME)

  ADD_UNIT_TEST(${exname} ${exfile})
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${exname} PRIVATE fmt::fmt)
endforeach(exfile)


