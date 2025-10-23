# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- modelling/costs : add getter and setter for cost term weights (https://github.com/Simple-Robotics/aligator/pull/359)
- support for Pinocchio 4 (https://github.com/Simple-Robotics/aligator/pull/361)
- added transparent hash op `ExtendedStringHash` (https://github.com/Simple-Robotics/aligator/pull/364)

### Changed

- move headers `allocator.hpp` and `arena-matrix.hpp` to `aligator/core` dir (https://github.com/Simple-Robotics/aligator/pull/362)
- python: aligator now requires eigenpy version 3.10.3 at least
- python: remove eigenpy version checks
- python: improve `CostStack.getComponent()` error message
- tests/python: more comprehensive test for `aligator.CostStack` (https://github.com/Simple-Robotics/aligator/pull/359)
- modelling/multibody : merge header `context.hpp` into `fwd.hpp`

### Removed

- modelling/multibody : remove txx files, remove `context.hpp`

## [0.16.0] - 2025-10-15

This release brings a major change to the ProxDDP solver, which no longer uses proximal iteration the co-state (the dynamics' Lagrange multiplies). This choice is made to increase the solver's overall performance.

The ProxDDP solver now defaults to using a linear rollout for the state-control trajectory update.

Furthermore, on a temporary basis, implicit discrete dynamics **are no longer supported** in the API and solvers.

One major new addition is the `ArenaMatrix` template class, which is a allocator-aware class which manages an Eigen matrix-like object and is compatible with Eigen operations. The design is inspired from [stan-dev/math](https://github.com/stan-dev/math)'s `arena_matrix` class.

### Fixed

- gar : fix missing move assignment operator in `LqrProblemTpl`
- Fix C++20 support. Fix `consteval` compilation errors related to fmt
- Fix `aligator::gar::ParallelRiccatiSolver` missing from docs
- Fix missing set of Pinocchio-support (cost, dynamics, etc) classes
- solver-proxddp : fix return type of `LinesearchVariant::isValid()`
- memory : added template class `ArenaMatrix` (https://github.com/Simple-Robotics/aligator/pull/348)

### Changed

- gar : allow setting number of refinement iterations for condensed KKT solver
- ProxDDP solver : change default rollout type to `RolloutType::LINEAR`/`ROLLOUT_LINEAR`
- solvers/proxddp : make `Workspace` an allocator-aware class
- add `[[nodiscard]]` attribute to several functions in:
  - core/manifold-base
  - gar/lqr-problem
  - gar/utils
  - modelling/spaces/cartesian-product
  - solvers/proxddp/solver-proxddp
- readme/cmake : update actually expected minimum version of eigenpy to 3.9
- solvers : make proxddp algo's Results class copyable again (in C++ and Python) (https://github.com/Simple-Robotics/aligator/pull/322)
- python/visitors : also set `__copy__` method on exposed class with `CopyableVisitor` (https://github.com/Simple-Robotics/aligator/pull/322)
- python : make `Results` copyable (using `CopyableVisitor`) (https://github.com/Simple-Robotics/aligator/pull/322)
- ProxDDP solver : remove proximal iteration/ALM over co-states
- ProxDDP solver : throw when given invalid linesearch enum value
- core/linesearches : move `LinesearchOptions` struct out of the `Linesearch` template class, add CTAD
- Change all tests to use Catch2 instead of Boost.Test
- The Riccati algorithms now run faster after the dual-regularisation on co-states and the QR for the implicit dynamics have been removed. The algos might be less numerically accurate.
  - the base interface and derived solvers no longer take the scalar argument `mudyn` (dual regularisation on the costate)
- Several classes are now allocator-aware:
  - the Riccati-based solvers in gar (`ProximalRiccatiSolver`, `ParallelRiccatiSolver`)
  - `WorkspaceTpl` in `aligator/solvers/proxddp`
- cmake/pixi: update maximum version for fmt library to v12 included (https://github.com/Simple-Robotics/aligator/pull/355)

#### Changes to dynamics

- Remove explicit dynamics (incl. explicit integrators) from `DynamicsModel` class hierarchy
- Make `ExplicitDynamicsModel` used everywhere in API (e.g. `StageModel` now takes/stores `polymorphic<ExplicitDynamicsModel>`)
- Directly store state space repr dim and actual dim (`nx` and `ndx`) in `ManifoldAbstract` class

### Added

- gar: add CTAD for the `ParallelRiccatiSolver` and `ProximalRiccatiSolver` classes
- testing: added a test_mpc.py script to test parallel and serial mpc implementations (https://github.com/Simple-Robotics/aligator/pull/331)
- modelling : added wheeled inverted pendulum dynamics (https://github.com/Simple-Robotics/aligator/pull/326)
- CMake option to `BUILD_STANDALONE_PYTHON_INTERFACE` (https://github.com/Simple-Robotics/aligator/pull/347)
- utils: add free function `forwardDynamics()` (replaces previous struct) in `aligator/utils/forward-dyn.hpp`

### Removed

- fwd.hpp : remove deprecated typedef `ODEDataTpl`
- gar: remove Cholmod backend ([#345](https://github.com/Simple-Robotics/aligator/pull/345))
- cmake / pixi : remove Cholmod dependency ([#345](https://github.com/Simple-Robotics/aligator/pull/345))
- gar: remove support for implicit dynamics in LQ solver interface
  - remove member `LqrKnotTpl::E`
  - simplify Riccati kernel algorithm
- memory : removed template class `ManagedMatrix` (https://github.com/Simple-Robotics/aligator/pull/348)
- removed several `.txx` template instantiation declaration files
- third-party : remove headers related to `boost::span`: `boost/core/data.hpp`, `boost/core/make_span.hpp`, `boost/core/span.hpp`
- removed header `aligator/tags.hpp`

## [0.15.0] - 2025-05-23

### Changed

- CMake : sync submodule (https://github.com/Simple-Robotics/aligator/pull/318)
- Update minimum required version of CMake to 3.22 (https://github.com/Simple-Robotics/aligator/pull/318)

## [0.14.1] - 2025-05-23

### Added

- Expose vs in ProxDDP result

## [0.14.0] - 2025-05-07

This release brings many major changes to aligator.

We officially **drop support for Pinocchio 2**. As we start looking forward to Pinocchio 4, aligator will require at least Pinocchio 3.4. If you need support for a lower version of Pinocchio 3, please contact us and suggest patches.

Furthermore, aligator will no longer depend on the proxsuite-nlp library, which will be archived in the near future - all functionality (manifolds, constraints) has been merged into aligator itself.

### Added

- Add allocator-aware class `ManagedMatrix` to wrap Eigen::Map types along with allocator-managed memory. Update `gar` classes to use `ManagedMatrix`. ([#313](https://github.com/Simple-Robotics/aligator/pull/313))
- Add CMake macro `aligator_create_python_extension()` to export ([#298](https://github.com/Simple-Robotics/aligator/pull/298))
- Add `LqrProblemTpl::isApprox()` and helper `lqrKnotsSameDim`, check dimensions in `LqrKnotTpl::isApprox()` ([#300](https://github.com/Simple-Robotics/aligator/pull/300))
- Add class `aligator::polymorphic_allocator` leveraging C++17 memory resources ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- Add memory allocator support for `LqrKnot` and `LqrProblem` ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- Add class `DenseKernel` for the stagewise-dense Riccati algo -- rework `DenseRiccatiSolver` class ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- Add `gar/fwd.hpp` header ([#301](https://github.com/Simple-Robotics/aligator/pull/301/))
- Add/transfer some headers from proxsuite-nlp (see [#309](https://github.com/Simple-Robotics/aligator/pull/309)):
  - Add `aligator/fmt-eigen.hpp` to use fmt with Eigen
  - Add `aligator/third-party/polymorphic_cxx14.h` ([#312](https://github.com/Simple-Robotics/aligator/pull/312))
  - Add Bunch-Kaufman solver, port from proxsuite-nlp in `aligator/bunchkaufman.hpp` ([#312](https://github.com/Simple-Robotics/aligator/pull/312))
  - Add linesearch classes `LinesearchBase` and `ArmijoLinesearch` ([#312](https://github.com/Simple-Robotics/aligator/pull/312))

### Changed

- Merge parts of proxsuite-nlp (manifolds and constraints) into aligator ([#309](https://github.com/Simple-Robotics/aligator/pull/309))
  - Remove dependency on proxsuite-nlp.
  - Make `aligator/math.hpp` independent: copy code for `math::check_value`, `math::check_scalar`, struct `math_types`.
  - Change `VerboseLevel` from a using-decl to its own enum in the `aligator` namespace.
  - Copy `ManifoldAbstractTpl` and subclasses over to aligator.
  - Copy `ConstraintSetTpl` and subclasses over to aligator, change using-decl in `fwd.hpp` to a proper fwd-declaration in the `aligator` namespace ([#310](https://github.com/Simple-Robotics/aligator/pull/310))
- `control-box-function.hpp`: update docstring, add deprecation dox tag ([#310](https://github.com/Simple-Robotics/aligator/pull/310))
- Rename `{ riccati-impl.hpp => riccati-kernel }` (and associated files) ([#301](https://github.com/Simple-Robotics/aligator/pull/301/))
- Move `HistoryCallbackTpl` from `aligator/helpers` (subdirectory was removed) to `aligator/core` ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- Move headers `results-base`, `solver-util`, `value-function`, `workspace-base` to `aligator/core` ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- `fwd.hpp`: do not include `<pinocchio/config.hpp>` header anymore ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- `blk-matrix.hpp`: remove protected ctors, add conversion ops ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- Headers `aligator/gar/*.hpp` merged into main library ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- CMake: rework declaration of examples & benchmarks (applying liberal use of `cmake_parse_arguments()`) ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- CMake: add `bench/CMakeLists.txt` to gersemi defs, apply formatting ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- CMake: apply compile definitions to `aligator` target directly (reduce use of global compile definitions) ([#243](https://github.com/Simple-Robotics/aligator/pull/243))

### Removed

- Removed `gar` as a separate CMake target and shared library, merge into main library ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- Remove function `allocate_shared_eigen_aligned()` ([#243](https://github.com/Simple-Robotics/aligator/pull/243))
- Officially remove support for Pinocchio 2, require Pinocchio >= 3.1 ([#307](https://github.com/Simple-Robotics/aligator/pull/307))
  - Remove the `ALIGATOR_PINOCCHIO_V3` compile definition.
- Remove macro header `aligator/macros.hpp` and macros `ALIGATOR_WITH_CPP_14`, `ALIGATOR_WITH_CPP_17`, and `ALIGATOR_MAYBE_UNUSED` ([#309](https://github.com/Simple-Robotics/aligator/pull/309))
- Remove deprecated typedef `aligator::context::CostBase` ([#310](https://github.com/Simple-Robotics/aligator/pull/310))
- Remove header `aligator/core/linesearch.hpp` (which using-decl linesearch classes from proxsuite-nlp) ([#312](https://github.com/Simple-Robotics/aligator/pull/312))
- Remove header `aligator/python/polymorphic-convertible.hpp` ([#309](https://github.com/Simple-Robotics/aligator/pull/309))

## [0.13.0] - 2025-04-26

### Added

- Add macro `ALIGATOR_OUT_OF_RANGE_ERROR` to throw `std::out_of_range` exceptions ([#294](https://github.com/Simple-Robotics/aligator/pull/294))

### Changed

- Rename `LQRKnotTpl` (C++)/`LqrKnot` (Python) to `LqrKnot(Tpl)` ([#283](https://github.com/Simple-Robotics/aligator/pull/283))
- Rename `LQRProblemTpl` (C++)/`LQRProblem` (Python) to `LqrProblem(Tpl)` ([#283](https://github.com/Simple-Robotics/aligator/pull/283))
- Reverse sign of StateErrorResidual ([#292](https://github.com/Simple-Robotics/aligator/pull/292))

### Fixed

- Correct references to paper's equations ([#284](https://github.com/Simple-Robotics/aligator/pull/284))
- Fix segfault in `FrameCollisionResidual`, instead throw `std::runtime_error` (**C++**, `RuntimeError` in **Python**) ([#294](https://github.com/Simple-Robotics/aligator/pull/294))

## [0.12.0] - 2025-03-27

### Changed

- Update for crocoddyl v3: boost -> std pointers ([#278](https://github.com/Simple-Robotics/aligator/issues/278))

## [0.11.0] - 2025-03-17

### Changed

- Only link against needed pinocchio libraries ([#260](https://github.com/Simple-Robotics/aligator/pull/260))
- Use Pinocchio instantiated functions ([#261](https://github.com/Simple-Robotics/aligator/pull/261))
- Link to pinocchio collision
- Some internal code now uses `TrajOptProblemTpl::initializeSolution()` to initialize state-control trajectories ([#274](https://github.com/Simple-Robotics/aligator/pull/274))
- Fix `HistoryCallback` init in examples ([#277](https://github.com/Simple-Robotics/aligator/pull/277))

### Fixed

- Fixed copy of TrajOptProblem ([#265](https://github.com/Simple-Robotics/aligator/pull/265))
- `LinesearchVariant::init()` should not be called unless the step accpetance strategy is a linesearch
- Fixed compilation issues with C++20 (resolving [#246](https://github.com/Simple-Robotics/aligator/issues/246) and [#254](https://github.com/Simple-Robotics/aligator/discussions/254))
- Prevent duplication of log columns ([#271](https://github.com/Simple-Robotics/aligator/pull/271))

### Added

- Add MPC test/example ([#272](https://github.com/Simple-Robotics/aligator/pull/272))
- Allow customization of the initial solution, introduce initialization strategies ([#274](https://github.com/Simple-Robotics/aligator/pull/274))
- Add a collision distance residual for collision pair
- Add a relaxed log-barrier cost function
- Add Nix support ([#268](https://github.com/Simple-Robotics/aligator/pull/268))

## [0.10.0] - 2024-12-09

### Added

- Add a multibody friction cone cost ([#234](https://github.com/Simple-Robotics/aligator/pull/234))
- Add a `GravityCompensationResidual`, modelling $r(x,u) = Bu - G(q)$ ([#235](https://github.com/Simple-Robotics/aligator/pull/235))
- Add Pixi support ([#240](https://github.com/Simple-Robotics/aligator/pull/240))
- Added a nonmonotone linesearch procedure ([#244](https://github.com/Simple-Robotics/aligator/pull/244))
- Add enum value `StepAcceptanceStrategy::LINESEARCH_NONMONOTONE` ([#244](https://github.com/Simple-Robotics/aligator/pull/244))

### Changed

- **API BREAKING:** Change enum value `StepAcceptanceStrategy::LINESEARCH` to `LINESEARCH_NONMONOTONE` ([#244](https://github.com/Simple-Robotics/aligator/pull/244))
  - Add constructor argument `StepAcceptanceStrategy sa_strategy`, defaults to nonmonotone
- The minimum required version of proxsuite-nlp is now 0.10.0 ([#244](https://github.com/Simple-Robotics/aligator/pull/244))
- `SolverProxDDP`: add temporary vectors for linesearch
- `SolverProxDDP`: remove exceptions from `computeMultipliers()`, return a bool flag
- HistoryCallback: take solver instance as argument
- `gar`: rework and move sparse matrix utilities to `gar/utils.hpp`
- `SolverProxDDP`: Rename `maxRefinementSteps_` and `refinementThreshold_` to snake-case
- `SolverProxDDP`: make `linesearch_` public
- Change uses of `ConstraintSetBase` template class to `ConstraintSetTpl` (following changes in proxsuite-nlp 0.9.0) ([#223](https://github.com/Simple-Robotics/aligator/pull/233))
- [gar] Throw an exception if trying to instantiate `ParallelRiccatiSolver` with num_threads smaller than 2.
- [API BREAKING] Rename friction cone for centroidal into CentroidalFrictionCone ([#234](https://github.com/Simple-Robotics/aligator/pull/234))
- Change the linear multibody friction cone to the true "ice cream" cone ([#238](https://github.com/Simple-Robotics/aligator/pull/238))
- [gar] Rework `RiccatiSolverDense` to not use inner struct `FactorData`
- Various changes to `gar` tests and `test_util`, add `LQRKnot::isApprox()`

### Removed

- Default constructor for `LQRProblemTpl`
- Removed header `gar/fwd.hpp` with forward-declarations

### Fixed

- Building aligator without Pinocchio support (including without Pinocchio support in the proxsuite-nlp dependency) is now supported. ([#250](https://github.com/Simple-Robotics/aligator/pull/250))

## [0.9.0] - 2024-10-11

### Added

- [python] Added getter `getComponent()` for `CostStack`, similar to C++ API
- Templated getters `getCost<U>()` and `getDynamics<U>()` in the StageModel class, and another `getDynamics<U>()` for integrator classes, to get the concrete types ([##205](https://github.com/Simple-Robotics/aligator/pull/205))
- Add a templated getter `getConstraint<U>` to ConstraintStack ([#222](https://github.com/Simple-Robotics/aligator/pull/222))
- python: Add helper `aligator.has_pinocchio_features()` ([#206](https://github.com/Simple-Robotics/aligator/pull/206))
- Add a cycleProblem function that properly rotates the problem data for MPC applications ([#215](https://github.com/Simple-Robotics/aligator/pull/215))
- Add a DCM cost function
- Add a cycleAppend function in Riccati solver headers to cycle the LQR solver
- `SolverProxDDP`: add manually settable dual feasibility tolerance `target_dual_tol_` with a getter and setter (with side effects)

### Changed

- All map types are now `boost::unordered_map` ([#203](https://github.com/Simple-Robotics/aligator/pull/203))
- Separate CostFiniteDifference out of finite-difference.hpp ([#212](https://github.com/Simple-Robotics/aligator/pull/212))
- Change the API of the wrench cost functions to allow 3D and 6D forces
- Separate centroidal wrench cone and multibody wrench cone costs
- Add a contact_name item to the CostMap structure
- Re-define ALM params struct internally to aligator ([#219](https://github.com/Simple-Robotics/aligator/pull/219))
- SolverProxDDP: add dynamics AL parameter scaling
- Rename `has_dyn_model` -> `hasDynModel` and `is_explicit` -> `isExplicit`
- Add `cost` (trajectory cost) column to logger
- `TrajOptProblem`: rename member `init_condition_` to `init_constraint_` (fitting with ctor argument name)
- python/utils: return axes instances from velocity/controls plotting helpers
- make `LinearFuntionTpl::evaluate()` call more efficient (using `.noalias()`)
- `HistoryCallbackTpl` now stores stored data directly
- Deprecate the `StageConstraintTpl` template struct, deprecate the related typedefs
- [python] Deprecate `aligator.StageConstraint`, functions and methods (e.g. `StageModel.addConstraint(cstr: StageConstraint)`) which use it
- Change formatting of exceptions, using variadic macro and type-erased implementation ([#230](https://github.com/Simple-Robotics/aligator/pull/230))

The following **API-BREAKING** changes come from PR [#229](https://github.com/Simple-Robotics/aligator/pull/229)

- Separate `DynamicsModelTpl` and its subclasses from the `StageFunctionTpl` class hierarchy
- Methods (`.evaluate()`, `.computeJacobians()`) in `StageFunctionTpl` are now binary (take only `(x, u)`)

### Removed

- Remove constraint scalers (including header `core/alm-weights.hpp`) from ProxDDP algorithm ([#214](https://github.com/Simple-Robotics/aligator/pull/214))
- SolverProxDDP: remove solver parameter `rho_` ([#221](https://github.com/Simple-Robotics/aligator/pull/221))
- Remove deprecated functions `ConstraintStackTpl::getDims` and `StageModelTpl::dyn_model`
- Remove `CallbackBaseTpl::post_linesearch_call(boost::any)`
- Remove unused headers `clone.hpp` and `version.hpp`

### Fixed

- Restore Pinocchio 3 Python tests ([#206](https://github.com/Simple-Robotics/aligator/pull/206))
- Fix warnings in `multibody-constraint-fwd.{hpp, hxx}` about deprecated Pinocchio types and functions (commit df04100c)

## [0.8.0] - 2024-09-18

### Added

- Getter `getResidual<Derived>()` for composite cost functions ([#198](https://github.com/Simple-Robotics/aligator/pull/198))
- Getter `getComponent<Derived>()` for `CostStack` ([#199](https://github.com/Simple-Robotics/aligator/pull/199))

### Changed

- Optimize a bunch of includes
- core: remove headers `proximal-penalty.hpp`/`proximal-penalty.hxx`
- Change storage of `CostStack` to `boost::unordered::unordered_map`, pointing to pair of cost function and weight ([#199](https://github.com/Simple-Robotics/aligator/pull/199))
- Change storage for `ConstraintStack` to using two `std::vector<polymorphic<>>` the struct `StageConstraintTpl` is now merely a convenient API shortcut for the end-user.
- Remove `StageConstraintTpl::nr()` (in C++ only)
- Update minimum required version of eigenpy to 3.7.0
- Add tracy macros to `stage-model.hxx`
- Change minimum required version of proxsuite-nlp to 0.8.0

## [0.7.0] - 2024-09-12

### Changed

- Use placement-new for `Workspace` and `Results` in solvers (FDDP and ProxDDP)
- Deprecate typedef for `std::vector<T, Eigen::aligned_allocator<T>>`
- Deprecate function template `allocate_shared_eigen_aligned<T>`
- Use custom macro defined in `aligator/tracy.hpp` to call Tracy ([#191](https://github.com/Simple-Robotics/aligator/pull/191))
- Change default behaviour with regards to Tracy (`DOWNLOAD_TRACY` is set to `OFF`)
- Upgrade minimum required version of proxsuite-nlp to 0.7.0

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

[Unreleased]: https://github.com/Simple-Robotics/aligator/compare/v0.16.0...HEAD
[0.16.0]: https://github.com/Simple-Robotics/aligator/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/Simple-Robotics/aligator/compare/v0.14.1...v0.15.0
[0.14.1]: https://github.com/Simple-Robotics/aligator/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/Simple-Robotics/aligator/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/Simple-Robotics/aligator/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/Simple-Robotics/aligator/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Simple-Robotics/aligator/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Simple-Robotics/aligator/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/Simple-Robotics/aligator/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Simple-Robotics/aligator/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Simple-Robotics/aligator/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Simple-Robotics/aligator/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Simple-Robotics/aligator/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/Simple-Robotics/aligator/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Simple-Robotics/aligator/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Simple-Robotics/aligator/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Simple-Robotics/aligator/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Simple-Robotics/aligator/releases/tag/v0.3.0
