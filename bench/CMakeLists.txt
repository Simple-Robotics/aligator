#
# Copyright (C) 2023 LAAS-CNRS, INRIA
#

# Create a benchmark
function(create_bench exfile croc)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "bench-${exname}")
  create_ex_or_bench(${exfile} ${exname})

  target_include_directories(${exname} PRIVATE ../examples)
  target_link_libraries(${exname} PRIVATE benchmark::benchmark)
  if(croc)
    target_link_libraries(${exname} PUBLIC aligator::croc_compat croc_talos_arm_utils)
  endif(croc)
endfunction()

function(create_gar_bench exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "bench-${exname}")
  add_executable(${exname} ${exfile})
  message(STATUS "Adding cpp bench ${exname}")
  set_target_properties(${exname} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${exname} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${exname} PRIVATE aligator::gar benchmark::benchmark gar_test_utils)
endfunction()

create_bench("lqr.cpp" FALSE)
create_bench("se2-car.cpp" FALSE)
if(BUILD_CROCODDYL_COMPAT)
  create_bench("croc-talos-arm.cpp" TRUE)
  target_add_example_robot_data(bench-croc-talos-arm)
endif()
if(PINOCCHIO_V3)
  create_bench("talos-walk.cpp" FALSE)
  target_add_example_robot_data(bench-talos-walk)
  target_add_talos_walk(bench-talos-walk)
endif()

create_gar_bench("gar.cpp")
