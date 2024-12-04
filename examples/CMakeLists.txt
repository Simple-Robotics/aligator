#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

cmake_policy(SET CMP0057 NEW)

# Create an example
macro(create_example exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "${PROJECT_NAME}-example-${exname}")
  create_ex_or_bench(False ${exfile} ${exname})
endmacro()

function(create_example_with_croco exfile)
  create_example(${exfile})
  target_link_libraries(
    ${exname}
    PRIVATE aligator::croc_compat croc_talos_arm_utils
  )
endfunction()

set(EXAMPLES_LIST clqr)

if(PROXSUITE_NLP_WITH_PINOCCHIO_SUPPORT)
  list(APPEND EXAMPLES_LIST se2-car)
endif()
if(PINOCCHIO_V3)
  list(APPEND EXAMPLES_LIST talos-walk)
endif()

foreach(filename ${EXAMPLES_LIST})
  create_example(${filename}.cpp)
endforeach()

if(PINOCCHIO_V3)
  target_add_talos_walk(aligator-example-talos-walk)
endif()

if(BUILD_CROCODDYL_COMPAT)
  create_example_with_croco(talos-arm.cpp)
  target_add_example_robot_data(aligator-example-talos-arm)
endif()

file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${pyfile}
      ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name}
  )
endforeach()
