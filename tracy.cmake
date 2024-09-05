# EXCLUDE_FROM_ALL removes all targets from the default "all" target.
# linking a target against TracyClient makes the latter a dependency, hence forces it to build.
# Net result: exclude Tracy listfile's installs from our own install/all target.
add_subdirectory(${TRACY_DIR} EXCLUDE_FROM_ALL)

set_target_properties(TracyClient PROPERTIES POSITION_INDEPENDENT_CODE True)

set(TRACY_STATIC
    ON
    CACHE INTERNAL ""
)
set(TRACY_ENABLE
    ${ENABLE_TRACY_PROFILING}
    CACHE INTERNAL ""
)
set(TRACY_PUBLIC_DIR ${PROJECT_SOURCE_DIR}/${TRACY_DIR}/public)
set(TRACY_HEADERS_INSTALL_DIR ${THIRD_PARTY_HEADERS_INSTALL_DIR})
message(STATUS "Tracy public dir: ${TRACY_PUBLIC_DIR}")

# List the tracy headers that our library exposes, so we can vendor them in the install.
set(tracy_includes ${TRACY_PUBLIC_DIR}/tracy/Tracy.hpp)
set(common_includes ${TRACY_PUBLIC_DIR}/common/TracyApi.h ${TRACY_PUBLIC_DIR}/common/TracyColor.hpp
                    ${TRACY_PUBLIC_DIR}/common/TracySystem.hpp
)

# Vendor the tracy headers - added to include interface later
install(FILES ${tracy_includes} DESTINATION ${TRACY_HEADERS_INSTALL_DIR}/tracy)
install(FILES ${common_includes} DESTINATION ${TRACY_HEADERS_INSTALL_DIR}/common)
