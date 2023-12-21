# aligator

**aligator** is a trajectory optimization library for robotics and beyond.

It can be used for motion generation and planning, for optimal estimation, for implementing Model-Predictive Control (MPC) schemes on complex systems, and more.

It contains ProxDDP, a primal-dual augmented Lagrangian-type trajectory optimization solver.

## Features

This is a C++ template library, which provides

* a modelling interface for optimal control problems, node-per-node
* an efficient solver algorithm for constrained OCPs
* (optional) support for the [pinocchio](https://github.com/stack-of-tasks/pinocchio) rigid-body dynamics library
* (optional) an interface to the [Crocoddyl](https://github.com/loco-3d/crocoddyl) trajectory optimization library which can be used as an alternative frontend
* Python bindings supported using [eigenpy](https://github.com/stack-of-tasks/eigenpy)

## Installation

### Dependencies

* [proxsuite-nlp](https://github.com/Simple-Robotics/proxsuite-nlp.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [Boost](https://www.boost.org) >= 1.71.0
* (optional) [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.1.0 | [conda](https://anaconda.org/conda-forge/eigenpy) (Python bindings)
* (optional) [Crocoddyl](https://github.com/loco-3d/crocoddyl)
* (optional) [Pinocchio](https://github.com/stack-of-tasks/eigenpy)
* a C++14 compliant compiler

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

* [Antoine Bambade](https://bambade.github.io/) (Inria): mathematics and algorithms developer
* [Justin Carpentier](https://jcarpent.github.io/) (Inria): project coordinator
* [Wilson Jallet](https://manifoldfr.github.io/) (LAAS-CNRS/Inria): main developer and manager of the project
* [Sarah Kazdadi](https://github.com/sarah-ek/): linear algebra czar
* [Quentin Le Lidec](https://quentinll.github.io/) (Inria)
* [Joris Vaillant](https://github.com/jorisv) (Inria): core developer
* [Nicolas Mansard](https://gepettoweb.laas.fr/index.php/Members/NicolasMansard) (LAAS-CNRS): project coordinator
* [Guilhem Saurel](https://github.com/nim65s) (LAAS-CNRS)
* [Fabian Schramm](https://github.com/fabinsch) (Inria): core developer
* [Ludovic De Matteïs](https://github.com/LudovicDeMatteis) (LAAS-CNRS/Inria)

## Acknowledgments

The development of **aligator** is actively supported by the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr) and the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr).

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the aligator project.
