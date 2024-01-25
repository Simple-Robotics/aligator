#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

cmake_policy(SET CMP0057 NEW)

# Create an example
macro(create_example exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "example-${exname}")
  create_ex_or_bench(${exfile} ${exname})
endmacro()

function(create_example_with_croco exfile)
  create_example(${exfile})
  target_link_libraries(${exname} PRIVATE aligator::croc_compat croc_talos_arm_utils)
endfunction()

create_example(se2-car.cpp)
create_example(clqr.cpp)

if(BUILD_CROCODDYL_COMPAT)
  create_example_with_croco(talos-arm.cpp)
  target_add_example_robot_data(example-talos-arm)
endif()

file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${pyfile}
                          ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name})
endforeach()
