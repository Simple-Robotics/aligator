#
# Copyright (C) 2023 LAAS-CNRS, INRIA
#

add_project_dependency(benchmark REQUIRED PKG_CONFIG_REQUIRES "benchmark >= 1.5.0")

# Create a benchmark
function(create_bench exfile croc)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "bench-${exname}")
  create_ex_or_bench(${exfile} ${exname})

  # include any headers in the examples/ directory
  target_include_directories(${exname} PRIVATE ${CMAKE_SOURCE_DIR}/examples)
  target_link_libraries(${exname} PRIVATE benchmark::benchmark)
  if(croc)
    target_link_libraries(${exname} PUBLIC aligator::croc_compat)
  endif(croc)
endfunction()

create_bench("lqr.cpp" FALSE)
if(BUILD_CROCODDYL_COMPAT)
  create_bench("croc-talos-arm.cpp" TRUE)
  target_add_example_robot_data(bench-croc-talos-arm)
  create_bench("multithreading.cpp" TRUE)
  target_add_example_robot_data(bench-multithreading)
endif()
