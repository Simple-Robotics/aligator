#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

ADD_PROJECT_DEPENDENCY(benchmark REQUIRED PKG_CONFIG_REQUIRES "benchmark >= 1.5.0")

file(GLOB EXAMPLES_CPP
     lqr.cpp)


foreach(exfile ${EXAMPLES_CPP})
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "example-${exname}")
  get_filename_component(exfile ${exfile} NAME)

  ADD_UNIT_TEST(${exname} ${exfile})
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${exname} PRIVATE benchmark::benchmark)
endforeach(exfile)

file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${pyfile}
    ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name}
  )
endforeach()
