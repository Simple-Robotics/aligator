#! /bin/bash
# Activation script

if [[ $PIXI_ENVIRONMENT_PLATFORMS == *"linux"* ]];
then
  # Conda compiler is named x86_64-conda-linux-gnu-c++, ccache can't resolve it
  # (https://ccache.dev/manual/latest.html#config_compiler_type)
  export CCACHE_COMPILERTYPE=gcc
fi

# Without -isystem, some LSP can't find headers
export ALIGATOR_CXX_FLAGS="$ALIGATOR_CXX_FLAGS -isystem $CONDA_PREFIX/include"

# Set default build value only if not previously set
export ALIGATOR_BUILD_TYPE=${ALIGATOR_BUILD_TYPE:=Release}
export ALIGATOR_PINOCCHIO_SUPPORT=${ALIGATOR_PINOCCHIO_SUPPORT:=OFF}
export ALIGATOR_CROCODDYL_COMPAT=${ALIGATOR_CROCODDYL_COMPAT:=OFF}
export ALIGATOR_OPENMP_SUPPORT=${ALIGATOR_OPENMP_SUPPORT:=OFF}
export ALIGATOR_BENCHMARKS=${ALIGATOR_BENCHMARKS:=ON}
export ALIGATOR_EXAMPLES=${ALIGATOR_EXAMPLES:=ON}
export ALIGATOR_PYTHON_STUBS=${ALIGATOR_PYTHON_STUBS:=ON}
export ALIGATOR_TRACY_ENABLE=${ALIGATOR_TRACY_ENABLE:=OFF}
