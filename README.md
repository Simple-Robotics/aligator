# proxddp

A primal-dual augmented Lagrangian-type trajectory optimization solver.

## Features

This package provides

* a modelling interface for optimal control problems, node-per-node
* an efficient solver algorithm

## Installation

### Dependencies

* [proxnlp](https://github.com/Simple-Robotics/proxnlp.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* Boost >= 1.71.0
* [eigenpy](https://github.com/stack-of-tasks/eigenpy) >= 2.7.2

### Build from source

```bash
git clone repo_link --recursive
# define envars here
cmake -DCMAKE_INSTALL_PREFIX=your_install_folder -S . -B build/
make -jNCPUS
```

For developers, add the `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` when working with language servers e.g. clangd.

**Building against conda:** define

```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
```

and use `$CONDA_PREFIX` as your install folder.
