# proxddp

A primal-dual augmented Lagrangian-type trajectory optimization solver.

## Features

This is a C++14 template library, which provides

* a modelling interface for optimal control problems, node-per-node
* an efficient solver algorithm
* (optional) an interface to [Crocoddyl](https://github.com/loco-3d/crocoddyl)

## Installation

### Dependencies

* [proxsuite-nlp](https://github.com/Simple-Robotics/proxsuite-nlp.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [Boost](https://www.boost.org) >= 1.71.0
* [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.1.0 | [conda](https://anaconda.org/conda-forge/eigenpy)
* (optional) [Crocoddyl](https://github.com/loco-3d/crocoddyl)

Python:

* [typed-argument-parser](https://github.com/swansonk14/typed-argument-parser)
* [matplotlib](https://matplotlib.org)

### Build from source

```bash
git clone repo_link --recursive
# define envars here
cmake -DCMAKE_INSTALL_PREFIX=your_install_folder -S . -B build/
cd build/
cmake --build . -jNCPUS
```

Options:

* For developers, add the `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` when working with language servers e.g. clangd.
* To use the Crocoddyl interface, add `-DBUILD_CROCODDYL_COMPAT=ON`
* By default, building the library will instantiate the templates for the `double` scalar type.

**Building against conda:** define

```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
```

and use `$CONDA_PREFIX` as your install folder.

## Benchmarking

```bash
./scripts/runbench.sh
```

For evaluating performance, we would recommend using [Flame Graphs](https://github.com/brendangregg/FlameGraph).
If you have the Rust toolchain and `cargo` installed, we suggest you install [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph). Then, you can create a flame graph with the following command:

```bash
flamegraph -o my_flamegraph.svg -- ./build/examples/example-croc-talos-arm
```

## Contributors

* [Antoine Bambade](https://bambade.github.io/)
* [Justin Carpentier](https://jcarpent.github.io/)
* [Wilson Jallet](https://manifoldfr.github.io/)
* [Sarah Kazdadi](https://github.com/sarah-ek/)
* [Quentin Le Lidec](https://quentinll.github.io/)
* [Nicolas Mansard](https://gepettoweb.laas.fr/index.php/Members/NicolasMansard)
* [Guilhem Saurel](https://github.com/nim65s)
* [Fabian Schramm](https://github.com/fabinsch)
* [Ludovic De Matteïs](https://github.com/LudovicDeMatteis)
