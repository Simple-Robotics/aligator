#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

cmake_policy(SET CMP0057 NEW)

# handle cpp examples
set(EXAMPLES_CPP "se2-car.cpp")
set(EXAMPLES_WITH_CROC_CPP "talos-arm.cpp")

find_package(example-robot-data 4.0.7 REQUIRED)
if(BUILD_CROCODDYL_COMPAT)
  find_package(crocoddyl REQUIRED)
  list(APPEND EXAMPLES_CPP talos-arm.cpp)
endif()

message(STATUS "Examples CPP: ${EXAMPLES_CPP}")

# Create an example
function(create_example exfile)
  get_filename_component(exname ${exfile} NAME_WE)
  set(exname "example-${exname}")
  create_ex_or_bench(${exfile} ${exname})

  if(${exfile} IN_LIST EXAMPLES_WITH_CROC_CPP)
    target_link_libraries(${exname} PUBLIC ${CROC_COMPAT_LIB_NAME})
  endif()
endfunction()

foreach(exfile ${EXAMPLES_CPP})
  create_example(${exfile})
endforeach(exfile)

if(BUILD_CROCODDYL_COMPAT)
  target_link_libraries(example-talos-arm PUBLIC proxnlp::cnpy)
endif()

file(GLOB EXAMPLES_PY *.py)

foreach(pyfile ${EXAMPLES_PY})
  get_filename_component(pyfile_name ${pyfile} NAME)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${pyfile}
                          ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name})
endforeach()
