# Aligator

**Aligator** is an efficient and versatile trajectory optimization library for robotics and beyond.

It can be used for motion generation and planning, optimal estimation, deployment of model-predictive control on complex systems, and much more.

## Features

**Aligator** is a C++ template library, which provides

* a modeling interface for optimal control problems, node-per-node
* a set of efficient solvers for constrained trajectory optimization
* support for the [pinocchio](https://github.com/stack-of-tasks/pinocchio) rigid-body dynamics library and its analytical derivatives
* an interface to the [Crocoddyl](https://github.com/loco-3d/crocoddyl) trajectory optimization library which can be used as an alternative frontend
* Python bindings leveraging [eigenpy](https://github.com/stack-of-tasks/eigenpy)

**Aligator** provides efficient implementations of the following algorithms for (constrained) trajectory optimization:

* ProxDDP: Proximal Differentiable Dynamic Programming, detailed in [this paper](https://inria.hal.science/hal-04332348/document)
* FeasibleDDP: Feasible Differentiable Dynamic Programming, detailed in [this paper](https://inria.hal.science/hal-02294059v1/document)

## Installation

### From Conda

From [our channel](https://anaconda.org/simple-robotics/aligator).

```bash
conda install -c simple-robotics aligator
```

### Build from source

```bash
git clone https://github.com/Simple-Robotics/aligator --recursive
cmake -DCMAKE_INSTALL_PREFIX=your_install_folder -S . -B build/ && cd build/
cmake --build . -jNCPUS
```

**Dependencies**

* [proxsuite-nlp](https://github.com/Simple-Robotics/proxsuite-nlp.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [Boost](https://www.boost.org) >= 1.71.0
* (optional) [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.1.0 | [conda](https://anaconda.org/conda-forge/eigenpy) (Python bindings)
* (optional) [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | [conda](https://anaconda.org/conda-forge/pinocchio)
* (optional) [Crocoddyl](https://github.com/loco-3d/crocoddyl) | [conda](https://anaconda.org/conda-forge/crocoddyl)
* (optional) [example-robot-data](https://github.com/Gepetto/example-robot-data) | [conda](https://anaconda.org/conda-forge/example-robot-data) (required for some examples and benchmarks)
* a C++14 compliant compiler

**Python dependencies**

* [typed-argument-parser](https://github.com/swansonk14/typed-argument-parser)
* [matplotlib](https://matplotlib.org)

### Notes

* For developers, add the `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` when working with language servers e.g. clangd.
* To use the Crocoddyl interface, add `-DBUILD_CROCODDYL_COMPAT=ON`
* By default, building the library will instantiate the templates for the `double` scalar type.
* To build against a Conda environment, activate the environment and run `export CMAKE_PREFIX_PATH=$CONDA_PREFIX` before running CMake and use `$CONDA_PREFIX` as your install folder.

## Benchmarking

We recommend using [Flame Graphs](https://github.com/brendangregg/FlameGraph) to evaluate performance.
If you have the Rust toolchain and `cargo` installed, we suggest you install [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph). Then, you can create a flame graph with the following command:

```bash
flamegraph -o my_flamegraph.svg -- ./build/examples/example-croc-talos-arm
```

## Citing Aligator

To cite **Aligator** in your academic research, please use the following bibtex entry:

```bibtex
@misc{aligatorweb,
  author = {Jallet, Wilson and Bambade, Antoine and El Kazdadi, Sarah and Justin, Carpentier and Nicolas, Mansard},
  title = {aligator},
  url = {https://github.com/Simple-Robotics/aligator}
}
```

## Contributors

* [Antoine Bambade](https://bambade.github.io/) (Inria): mathematics and algorithms developer
* [Justin Carpentier](https://jcarpent.github.io/) (Inria): project instructor
* [Wilson Jallet](https://manifoldfr.github.io/) (LAAS-CNRS/Inria): main developer and manager of the project
* [Sarah Kazdadi](https://github.com/sarah-ek/): linear algebra czar
* [Quentin Le Lidec](https://quentinll.github.io/) (Inria): feature developer
* [Joris Vaillant](https://github.com/jorisv) (Inria): core developer
* [Nicolas Mansard](https://gepettoweb.laas.fr/index.php/Members/NicolasMansard) (LAAS-CNRS): project coordinator
* [Guilhem Saurel](https://github.com/nim65s) (LAAS-CNRS): core maintener
* [Fabian Schramm](https://github.com/fabinsch) (Inria): core developer
* [Ludovic De Matteïs](https://github.com/LudovicDeMatteis) (LAAS-CNRS/Inria): feature developer

## Acknowledgments

The development of **Aligator** is actively supported by the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr) and the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr).
