add_library(aligator_third_party INTERFACE)

# Install third-party library to aligator's installed include and lib dirs.
function(aligator_install_third_party target)
  set(
    THIRD_PARTY_INSTALL_INCLUDEDIR
    ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/third-party
  )
  cmake_parse_arguments(
    "EXT"
    ""
    "INCLUDE_DIR;HEADER_DESTINATION"
    "HEADER_FILES"
    ${ARGN}
  )
  install(
    TARGETS ${target}
    EXPORT ${TARGETS_EXPORT_NAME}
    PUBLIC_HEADER DESTINATION ${THIRD_PARTY_INSTALL_INCLUDEDIR}
  )
  if(EXT_INCLUDE_DIR)
    install(
      DIRECTORY ${EXT_INCLUDE_DIR}
      DESTINATION ${THIRD_PARTY_INSTALL_INCLUDEDIR}
      FILES_MATCHING
      REGEX ".*\\.(h|hpp|hxx)$"
    )
  endif()
  # install individual header files
  if(EXT_HEADER_FILES)
    foreach(file ${EXT_HEADER_FILES})
      install(
        FILES ${file}
        DESTINATION ${THIRD_PARTY_INSTALL_INCLUDEDIR}/${EXT_HEADER_DESTINATION}
      )
    endforeach()
  endif()
endfunction()

add_library(expected INTERFACE)
target_include_directories(
  expected
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/expected/include>
)
aligator_install_third_party(expected INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/expected/include/tl
)
target_link_libraries(aligator_third_party INTERFACE expected)

install(TARGETS aligator_third_party EXPORT ${TARGETS_EXPORT_NAME})
