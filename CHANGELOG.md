# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fix finite difference function in python unittest ([#128](https://github.com/Simple-Robotics/aligator/pull/128))

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

[Unreleased]: https://github.com/Simple-Robotics/aligator/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Simple-Robotics/aligator/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Simple-Robotics/aligator/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Simple-Robotics/aligator/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Simple-Robotics/aligator/releases/tag/v0.3.0
