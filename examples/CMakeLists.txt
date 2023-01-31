#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

ADD_PROJECT_DEPENDENCY(benchmark REQUIRED PKG_CONFIG_REQUIRES "benchmark >= 1.5.0")

cmake_policy(SET CMP0057 NEW)

# handle cpp examples
set(EXAMPLES_CPP
    "se2-car.cpp")

set(BENCHMARKS_CPP
    "bench/lqr.cpp"
    "bench/croc-talos-arm.cpp")

set(EXAMPLES_WITH_CROC_CPP
    "bench/croc-talos-arm.cpp"
    "talos-arm.cpp"
    )

list(APPEND EXAMPLES_CPP ${BENCHMARKS_CPP})
list(APPEND EXAMPLES_CPP ${EXAMPLES_WITH_CROC_CPP})
list(REMOVE_DUPLICATES EXAMPLES_CPP)

message(STATUS "Examples CPP: ${EXAMPLES_CPP}")

foreach(exfile ${EXAMPLES_CPP})
  get_filename_component(exname ${exfile} NAME_WE)
  if (${exfile} IN_LIST BENCHMARKS_CPP)
    set(exname "bench-${exname}")
  else()
    set(exname "example-${exname}")
  endif()

  ADD_UNIT_TEST(${exname} ${exfile})
  message(STATUS "Adding cpp example ${exname}")
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} ${PROJECT_NAME})

  if (${exfile} IN_LIST BENCHMARKS_CPP)
    target_link_libraries(${exname} benchmark::benchmark)
  endif()

  if (${exfile} IN_LIST EXAMPLES_WITH_CROC_CPP)
    ADD_PROJECT_DEPENDENCY(crocoddyl REQUIRED)
    target_link_libraries(${exname} ${CROC_COMPAT_LIB_NAME})
  endif()
endforeach(exfile)

target_link_libraries(example-talos-arm proxnlp::cnpy)


file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${pyfile}
    ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name}
  )
endforeach()
