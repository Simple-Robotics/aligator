/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <pinocchio/multibody/fwd.hpp>

#include <crocoddyl/core/fwd.hpp>

namespace pin = pinocchio;
namespace croc = crocoddyl;

void makeTalosArm(pin::Model &model);

/// This reimplements the Crocoddyl problem defined in
/// examples/croc_arm_manipulation.py.
std::shared_ptr<croc::ShootingProblem>
defineCrocoddylProblem(std::size_t nsteps);

void getInitialGuesses(
    const std::shared_ptr<croc::ShootingProblem> &croc_problem,
    std::vector<Eigen::VectorXd> &xs_i, std::vector<Eigen::VectorXd> &us_i);
