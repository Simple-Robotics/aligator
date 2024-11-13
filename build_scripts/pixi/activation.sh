#! /bin/bash
# Activation script

# Remove flags setup from cxx-compiler
unset CFLAGS
unset CPPFLAGS
unset CXXFLAGS
unset DEBUG_CFLAGS
unset DEBUG_CPPFLAGS
unset DEBUG_CXXFLAGS
unset LDFLAGS

if [[ $host_alias == *"apple"* ]];
then
  # On OSX setting the rpath and -L it's important to use the conda libc++ instead of the system one.
  # If conda-forge use install_name_tool to package some libs, -headerpad_max_install_names is then mandatory
  export LDFLAGS="-Wl,-headerpad_max_install_names -Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
elif [[ $host_alias == *"linux"* ]];
then
  # On GNU/Linux, I don't know if these flags are mandatory with g++ but
  # it allow to use clang++ as compiler
  export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath-link,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
fi

# Setup ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Create compile_commands.json for language server
export CMAKE_EXPORT_COMPILE_COMMANDS=1

# Activate color output with Ninja
export CMAKE_COLOR_DIAGNOSTICS=1

# Set default build value only if not previously set
export ALIGATOR_BUILD_TYPE=${ALIGATOR_BUILD_TYPE:=Release}
export ALIGATOR_PINOCCHIO_SUPPORT=${ALIGATOR_PINOCCHIO_SUPPORT:=OFF}
export ALIGATOR_CROCODDYL_COMPAT=${ALIGATOR_CROCODDYL_COMPAT:=OFF}
export ALIGATOR_OPENMP_SUPPORT=${ALIGATOR_OPENMP_SUPPORT:=OFF}
export ALIGATOR_CHOLMOD_SUPPORT=${ALIGATOR_CHOLMOD_SUPPORT:=OFF}
export ALIGATOR_BENCHMARK=${ALIGATOR_BENCHMARK:=ON}
export ALIGATOR_EXAMPLES=${ALIGATOR_EXAMPLES:=ON}
export ALIGATOR_PYTHON_STUBS=${ALIGATOR_PYTHON_STUBS:=ON}

# Bugfix:
# Use BoostConfig module distributed by boost library instead of using FindBoost module distributed
# by CMake.
# But find_package(SDFormat14) reset this policy (with cmake_minimum_required) and redefine the find_dependency macro.
# Then, futur call of this macro don't use the right version of this policy.
export CMAKE_POLICY_DEFAULT_CMP0167=NEW
