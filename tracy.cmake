# EXCLUDE_FROM_ALL removes all targets from the default "all" target.
# linking a target against TracyClient makes the latter a dependency, hence forces it to build.
# Net result: exclude Tracy listfile's installs from our own install/all target.
# add_subdirectory(${TRACY_DIR} EXCLUDE_FROM_ALL)

# set_target_properties(TracyClient PROPERTIES POSITION_INDEPENDENT_CODE True)
find_package(Threads REQUIRED)

set(TRACY_PUBLIC_DIR ${PROJECT_SOURCE_DIR}/${TRACY_DIR}/public)
add_library(aligator_profiling SHARED ${TRACY_PUBLIC_DIR}/TracyClient.cpp)
add_library(aligator::profiling ALIAS aligator_profiling)
set_standard_output_directory(aligator_profiling)
set_target_properties(
  aligator_profiling
  PROPERTIES LINKER_LANGUAGE CXX
             VERSION ${PROJECT_VERSION}
             INSTALL_RPATH "\$ORIGIN"
)

target_compile_definitions(aligator_profiling PUBLIC TRACY_ENABLE=${ENABLE_TRACY_PROFILING})
target_compile_definitions(aligator_profiling PRIVATE TRACY_EXPORTS)
target_compile_definitions(aligator_profiling PUBLIC TRACY_IMPORTS)
target_include_directories(
  aligator_profiling SYSTEM PUBLIC $<BUILD_INTERFACE:${TRACY_PUBLIC_DIR}> $<INSTALL_INTERFACE:include>
)
target_link_libraries(aligator_profiling PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

macro(set_option option value)
  if(${option})
    message(STATUS "${option}: ON")
    target_compile_definitions(aligator_profiling PUBLIC ${option})
  elseif()
    message(STATUS "${option}: OFF")
  endif()
endmacro()

set(TRACY_HEADERS_INSTALL_DIR ${THIRD_PARTY_HEADERS_INSTALL_DIR})
message(STATUS "Tracy public dir: ${TRACY_PUBLIC_DIR}")

# List the tracy headers that our library exposes, so we can vendor them in the install.
set(tracy_includes ${TRACY_PUBLIC_DIR}/tracy/Tracy.hpp)
set(client_includes
    ${TRACY_PUBLIC_DIR}/client/tracy_concurrentqueue.h
    ${TRACY_PUBLIC_DIR}/client/tracy_rpmalloc.hpp
    ${TRACY_PUBLIC_DIR}/client/tracy_SPSCQueue.h
    ${TRACY_PUBLIC_DIR}/client/TracyKCore.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyArmCpuTable.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyCallstack.h
    ${TRACY_PUBLIC_DIR}/client/TracyCallstack.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyCpuid.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyDebug.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyDxt1.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyFastVector.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyLock.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyProfiler.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyRingBuffer.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyScoped.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyStringHelpers.hpp
    ${TRACY_PUBLIC_DIR}/client/TracySysPower.hpp
    ${TRACY_PUBLIC_DIR}/client/TracySysTime.hpp
    ${TRACY_PUBLIC_DIR}/client/TracySysTrace.hpp
    ${TRACY_PUBLIC_DIR}/client/TracyThread.hpp
)
set(common_includes
    ${TRACY_PUBLIC_DIR}/common/tracy_lz4.hpp
    ${TRACY_PUBLIC_DIR}/common/tracy_lz4hc.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyAlign.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyAlloc.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyApi.h
    ${TRACY_PUBLIC_DIR}/common/TracyColor.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyForceInline.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyMutex.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyProtocol.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyQueue.hpp
    ${TRACY_PUBLIC_DIR}/common/TracySocket.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyStackFrames.hpp
    ${TRACY_PUBLIC_DIR}/common/TracySystem.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyUwp.hpp
    ${TRACY_PUBLIC_DIR}/common/TracyYield.hpp
)

# Copy tracy headers to the build tree
foreach(headerFile IN LISTS tracy_includes client_includes common_includes)
  string(REPLACE "${TRACY_PUBLIC_DIR}/" "" headerFileRelative ${headerFile})
  set(headerCopyPath include/${THIRD_PARTY_HEADERS_DIR}/${headerFileRelative})
  message(DEBUG "Copy header ${headerFile} to build tree at path <build-dir>/${headerCopyPath}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${headerFile} ${PROJECT_BINARY_DIR}/${headerCopyPath})
endforeach()

# Vendor the tracy headers - added to include interface later
install(FILES ${tracy_includes} DESTINATION ${TRACY_HEADERS_INSTALL_DIR}/tracy)
install(FILES ${client_includes} DESTINATION ${TRACY_HEADERS_INSTALL_DIR}/client)
install(FILES ${common_includes} DESTINATION ${TRACY_HEADERS_INSTALL_DIR}/common)
