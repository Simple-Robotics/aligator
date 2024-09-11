# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Use placement-new for `Workspace` and `Results` in solvers (FDDP and ProxDDP)
- Deprecate typedef for `std::vector<T, Eigen::aligned_allocator<T>>`
- Deprecate function template `allocate_shared_eigen_aligned<T>`

### Fixed

- Fix RiccatiSolverDense initialization ([#174](https://github.com/Simple-Robotics/aligator/pull/174))
- Remove CMake CMP0167 and CMP0169 warnings ([#176](https://github.com/Simple-Robotics/aligator/pull/176))

### Added

- Add compatibility with jrl-cmakemodules workspace ([#172](https://github.com/Simple-Robotics/aligator/pull/172))

## [0.6.1] - 2024-05-27

### Added

- Add force and wrench cone costs
- Add centroidal momentum cost

### Changed

- Do not compile or use `gar::ParallelRiccatiSolver<>` when OpenMP support is disabled ([#160](https://github.com/Simple-Robotics/aligator/pull/160))
- Allow to build with fmt 11 ([#173](https://github.com/Simple-Robotics/aligator/pull/173))


## [0.6.0] - 2024-05-23

### Added
- Added constrained LQR example ([#145](https://github.com/Simple-Robotics/aligator/pull/145))
- Adds [tracy](https://github.com/Simple-Robotics/tracy) as a profiling tool
- Adds a new sublibrary called `gar` to represent and solve time-varying linear-quadratic subproblems
- Adds a parallel, block-sparse factorization for the implicit/proximal Riccati algorithm
- Integrates the CHOLMOD solver from the SuiteSparse library into `gar`
- Add a C++ example and benchmark of Talos walking by @edantec
- Add a `BlkMatrix<>` template class for dealing with block Eigen matrices effectively
- Copy the headers from [boost::core::span](https://www.boost.org/doc/libs/1_85_0/libs/core/doc/html/core/span.html)
- Add SE2 car benchmark `bench/se2-car.cpp`
- Split part of `macros.hpp` into new header `eigen-macros.hpp`, add `ALIGATOR_NOMALLOC_SCOPED` macro to disable Eigen's malloc per-scope and a caching system to restore the malloc status
- Add `context.hpp` file for `aligator/modelling/dynamics`

### Changed

- Standardized CMake output directories ([#147](https://github.com/Simple-Robotics/aligator/pull/147))
- Split derivative computation into first- and second-order functions per stage by @fabinsch
- Changed minimum version of C++ to C++17 (no longer use `boost::optional` or `boost::filesystem`)
- SolverProxDDP now uses linear solvers provided by `gar` (API breaking), add `LQSolverChoice` enum
- Minimum version of eigenpy upgraded to 3.4 (for supporting `std::unique_ptr`)
- Move cost functions to `aligator/modelling/costs`
- Deprecate `ControlBoxFunction`
- Remove `LDLTChoice` enum (API breaking)
- Refactor computation of problem Lagrangian's gradient
- Remove `aligator/core/stage-data.hxx`
- StageModel/StageData now store dynamics model/data separate from other constraints, add `dynamics_data` to `StageData`
- Rewrite the `Logger` class (rename from `BaseLogger`) using map data structures to store line formatting info + content
- Merge struct ODEData into ContinousDynamicsData, rename continous-base.hpp header to continuous-dynamics-abstract
- Optimize a few includes for faster compilation:
  - `aligator/modelling/dynamics/fwd.hpp`
  - `aligator/core/stage-model.hpp` no longer includes `aligator/core/cost-abstract.hpp`
  - Split `traj-opt-data.hpp` out of `traj-opt-problem.hpp`

#### Python API

- Rename `ContinousDynamicsBase` to `ContinousDynamicsAbstract`
- Rename `CostBase` to `CostAbstract`
- Expose `TrajOptData.init_data`
- Remove `LDLTChoice` enum and `SolverProxDDP.ldlt_solver_choice` attribute

## [0.5.1] - 2024-04-24

### Fixed
- Fix finite difference function in python unittest ([#128](https://github.com/Simple-Robotics/aligator/pull/128))

### Added
- Add kinodynamics forward scheme
- Add centroidal dynamics derivative cost to regularize the time derivative of centroidal dynamics
- Add Python example of Solo stepping in place
- Add wrench cone cost for 6D contact in centroidal dynamics
- Add a 6D contact formulation for every centroidal cost and dynamics, including kinodynamics

## [0.5.0] - 2024-02-13

### Added

- Add a pair filter strategy as an alternative to linesearch methods
- Add a python example of a locomotion OCP with the robot Talos
- Add two nonlinear centroidal dynamical models, where the control is respectively contact forces and their time derivative
- Add a set of cost functions to generate a centroidal problem with user-defined contacts
- Add unittests and example of typical centroidal problem

## [0.4.1] - 2024-01-25

### Changed

- CMake: fetch submodule if not available in ([#103](https://github.com/Simple-Robotics/aligator/pull/103))
- CMake: move benchmark dependency speecification in main file
- Document and simplify the LQR example
- Finish the se2-car.cpp example ([#110](https://github.com/Simple-Robotics/aligator/pull/110))
- Add template instantiation for IntegratorAbstract, ExplicitIntegratorAbstract and IntegratorRK2 ([#114](https://github.com/Simple-Robotics/aligator/pull/114))
- Don't output numpy matrices in `example-talos-arm`

### Fixed

- Fix name of frame parent attribute in examples
- Export C++ definitions in CMake config file
- Fix Doxyfile python bindings directory ([#110](https://github.com/Simple-Robotics/aligator/pull/110))
- Fix for eigenpy 3.3 ([#121](https://github.com/Simple-Robotics/aligator/pull/121))

## [0.4.0] - 2023-12-22

### Added

* This is the first release of `aligator`. This library is a joint effort between INRIA and LAAS-CNRS, and will be maintained and expanded in the future. Please provide constructive feedback and contribute!

[Unreleased]: https://github.com/Simple-Robotics/aligator/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/Simple-Robotics/aligator/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Simple-Robotics/aligator/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/Simple-Robotics/aligator/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Simple-Robotics/aligator/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Simple-Robotics/aligator/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Simple-Robotics/aligator/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Simple-Robotics/aligator/releases/tag/v0.3.0
