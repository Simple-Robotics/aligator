#! /bin/sh

rm -rf build

mkdir build
cd build

cmake ${CMAKE_ARGS} .. \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$PYTHON \
      -DBUILD_PYTHON_INTERFACE=ON \
      -DGENERATE_PYTHON_STUBS=ON \
      -DBUILD_WITH_PINOCCHIO_SUPPORT=ON \
      -DBUILD_CROCODDYL_COMPAT=ON \
      -DBUILD_WITH_OPENMP_SUPPORT=ON \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTING=OFF

ninja
ninja install
