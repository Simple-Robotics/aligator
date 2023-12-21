#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

cmake_policy(SET CMP0057 NEW)

# handle cpp examples
set(EXAMPLES_WITH_CROC_CPP "talos-arm.cpp")

find_package(example-robot-data 4.0.7 REQUIRED)
if(BUILD_CROCODDYL_COMPAT)
  find_package(crocoddyl REQUIRED)
endif()

# Create an example
macro(create_example exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "example-${exname}")
  create_ex_or_bench(${exfile} ${exname})
endmacro()

function(create_example_with_croco exfile)
  create_example(${exfile})
  target_link_libraries(${exname} PUBLIC aligator::croc_compat)
endfunction()

create_example(se2-car.cpp)

if(BUILD_CROCODDYL_COMPAT)
  create_example_with_croco(talos-arm.cpp)
endif()

if(BUILD_CROCODDYL_COMPAT)
  target_link_libraries(example-talos-arm PUBLIC proxsuite-nlp::cnpy)
endif()

file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${pyfile}
                          ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name})
endforeach()
