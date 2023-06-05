#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

add_project_dependency(benchmark REQUIRED PKG_CONFIG_REQUIRES "benchmark >= 1.5.0")

cmake_policy(SET CMP0057 NEW)

# handle cpp examples
set(EXAMPLES_CPP "se2-car.cpp")
set(BENCHMARKS_CPP "bench/lqr.cpp")

set(EXAMPLES_WITH_CROC_CPP "bench/croc-talos-arm.cpp" "bench/multithreading.cpp" "talos-arm.cpp")

message(STATUS "Crocoddyl support: ${BUILD_CROCODDYL_COMPAT}")
if(BUILD_CROCODDYL_COMPAT)
  find_package(crocoddyl REQUIRED)
  find_package(example-robot-data REQUIRED)
  list(APPEND BENCHMARKS_CPP bench/croc-talos-arm.cpp bench/multithreading.cpp)
  list(APPEND EXAMPLES_CPP talos-arm.cpp)
endif()

message(STATUS "Examples CPP: ${EXAMPLES_CPP}")
message(STATUS "Benchmarks CPP: ${BENCHMARKS_CPP}")

macro(create_ex_or_bench exfile exname)

  add_unit_test(${exname} ${exfile})
  message(STATUS "Adding cpp example ${exname}")
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${exname} PUBLIC example-robot-data::example-robot-data)
  if(BUILD_WITH_OPENMP_SUPPORT)
    target_link_libraries(${exname} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endmacro()

# Create an example
function(create_example exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "example-${exname}")
  create_ex_or_bench(${exfile} ${exname})

  if(${exfile} IN_LIST EXAMPLES_WITH_CROC_CPP)
    target_link_libraries(${exname} PUBLIC ${CROC_COMPAT_LIB_NAME})
  endif()
endfunction()

# Create a benchmark
function(create_bench exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "bench-${exname}")
  create_ex_or_bench(${exfile} ${exname})

  target_link_libraries(${exname} PRIVATE benchmark::benchmark)

  if(${exfile} IN_LIST EXAMPLES_WITH_CROC_CPP)
    target_link_libraries(${exname} PUBLIC ${CROC_COMPAT_LIB_NAME})
  endif()
endfunction()

foreach(exfile ${EXAMPLES_CPP})
  create_example(${exfile})
endforeach(exfile)

foreach(exfile ${BENCHMARKS_CPP})
  create_bench(${exfile})
endforeach(exfile)

if(BUILD_CROCODDYL_COMPAT)
  target_link_libraries(example-talos-arm PUBLIC proxnlp::cnpy)
  target_link_libraries(example-talos-arm PUBLIC example-robot-data::example-robot-data)
endif()

file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${pyfile}
                          ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name})
endforeach()
