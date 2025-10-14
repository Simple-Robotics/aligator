# ------
# aligator_add_python_extension
#
#
# ------
function(aligator_create_python_extension name)
  message(STATUS "Adding aligator Python C++ extension ${name}")
  set(options WITH_SOABI)
  cmake_parse_arguments(arg "${options}" "" "" ${ARGN})
  set(_parse_oneValueArgs)
  foreach(op IN LISTS options)
    if(${arg_${op}})
      list(APPEND _parse_oneValueArgs ${op})
    endif()
  endforeach()
  set(_sources ${arg_UNPARSED_ARGUMENTS})

  Python3_add_library(${name} ${_parse_oneValueArgs} ${_sources})
  target_link_libraries(${name} PRIVATE eigenpy::eigenpy aligator::aligator)
  target_compile_definitions(${name} PRIVATE PYTHON_MODULE_NAME=${name})
endfunction()
