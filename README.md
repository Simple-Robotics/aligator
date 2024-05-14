# Aligator

<a href="https://simple-robotics.github.io/aligator/"><img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Documentation"/></a>

**Aligator** is an efficient and versatile trajectory optimization library for robotics and beyond.

It can be used for motion generation and planning, optimal estimation, deployment of model-predictive control on complex systems, and much more.

Developing advanced, open-source, and versatile robotic software such as **Aligator** takes time and energy while requiring a lot of engineering support.
In recognition of our commitment, we would be grateful if you would quote our papers and software in your publications, software, and research articles.
Please refer to the [Citation section](#citing-aligator) for further details.

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

From either conda-forge or [our channel](https://anaconda.org/simple-robotics/aligator).

```bash
conda install -c conda-forge aligator  # or -c conda-forge
```

### Build from source

```bash
git clone https://github.com/Simple-Robotics/aligator --recursive
cmake -DCMAKE_INSTALL_PREFIX=your_install_folder -S . -B build/ && cd build/
cmake --build . -jNCPUS
```

#### Dependencies

* [proxsuite-nlp](https://github.com/Simple-Robotics/proxsuite-nlp.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [Boost](https://www.boost.org) >= 1.71.0
* OpenMP
* (optional) [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.4.0 | [conda](https://anaconda.org/conda-forge/eigenpy) (Python bindings)
* (optional) [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | [conda](https://anaconda.org/conda-forge/pinocchio)
* (optional) [Crocoddyl](https://github.com/loco-3d/crocoddyl) | [conda](https://anaconda.org/conda-forge/crocoddyl)
* (optional) [example-robot-data](https://github.com/Gepetto/example-robot-data) | [conda](https://anaconda.org/conda-forge/example-robot-data) (required for some examples and benchmarks)
* a C++17 compliant compiler

#### Python dependencies

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
Please also consider citing the reference paper for the ProxDDP algorithm:

```bibtex
@misc{jalletPROXDDPProximalConstrained2023,
  title = {{PROXDDP: Proximal Constrained Trajectory Optimization}},
  author = {Jallet, Wilson and Bambade, Antoine and Arlaud, Etienne and {El-Kazdadi}, Sarah and Mansard, Nicolas and Carpentier, Justin},
  year = {2023},
  abstract = {Trajectory optimization (TO) has proven, over the last decade, to be a versatile and effective framework for robot control. Several numerical solvers have been demonstrated to be fast enough to allow recomputing full-dynamics trajectories for various systems at control time, enabling model predictive control (MPC) of complex robots. These first implementations of MPC in robotics predominantly utilize some differential dynamic programming (DDP) variant for its computational speed and ease of use in constraint-free settings. Nevertheless, many scenarios in robotics call for adding hard constraints in TO problems (e.g., torque limits, obstacle avoidance), which existing solvers, based on DDP, often struggle to handle. Effectively addressing path constraints still poses optimization challenges (e.g., numerical stability, efficiency, accuracy of constraint satisfaction) that we propose to solve by combining advances in numerical optimization with the foundational efficiency of DDP. In this article, we leverage proximal methods for constrained optimization and introduce a DDP-like method to achieve fast, constrained trajectory optimization with an efficient warm-starting strategy particularly suited for MPC applications. Compared to earlier solvers, our approach effectively manages hard constraints without warm-start limitations and exhibits commendable convergence accuracy. Additionally, we leverage the computational efficiency of DDP, enabling real-time resolution of complex problems such as whole-body quadruped locomotion. We provide a complete implementation as part of an open-source and flexible C++ trajectory optimization library called ALIGATOR. These algorithmic contributions are validated through several trajectory planning scenarios from the robotics literature and the real-time whole-body MPC of a quadruped robot.},
  langid = {english},
  note = {https://inria.hal.science/hal-04332348v1}
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
* [Guilhem Saurel](https://github.com/nim65s) (LAAS-CNRS): core maintainer
* [Fabian Schramm](https://github.com/fabinsch) (Inria): core developer
* [Ludovic De Matteïs](https://github.com/LudovicDeMatteis) (LAAS-CNRS/Inria): feature developer
* [Ewen Dantec](https://edantec.github.io/) (Inria): feature developer

## Acknowledgments

The development of **Aligator** is actively supported by the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr) and the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr).
